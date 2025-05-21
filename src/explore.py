# The `DataExplorer` class provides methods for exploratory data analysis and visualization on a
# preprocessed sales dataset, offering insights on delay rates, trends, correlations, and more.


# Importing several libraries commonly used for data analysis and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataExplorer:
    def __init__(self, df):
        """
        Initialize the explorer with a cleaned DataFrame
        :param df: pd.DataFrame - preprocessed sales dataset
        """
        self.df = df

    def get_kpis(self):
        """
        Return summary KPIs: total orders, delay rate, average lead time, average delay days
        Returns: dict
        """
        return {
            "total_orders": len(self.df),
            "delay_rate": round(self.df['delay_flag'].mean() * 100, 1),
            "avg_lead_time": round(self.df['lead_time'].mean(), 1),
            "avg_delay_days": round(self.df[self.df['delay_flag'] == 1]['delay_days'].mean(), 1)
        }

    def get_grouped_delay_rate(self, group_cols):
        """
        Calculate delay rate grouped by one or more columns
        :param group_cols: list[str] - list of columns to group by
        Returns: pd.DataFrame
        """
        return self.df.groupby(group_cols)['delay_flag'].mean().reset_index().sort_values('delay_flag', ascending=False)

    def get_monthly_delay_trends(self):
        """
        Return delay rate as a pivot table of year vs month
        Returns: pd.DataFrame
        """
        return self.df.pivot_table(values='delay_flag', index='year', columns='month', aggfunc='mean')

    def get_delay_reason_counts(self):
        """
        Return counts of delay reasons from ECD Notes
        Returns: pd.Series
        """
        return self.df['ECD Notes'].value_counts()

    def get_correlation_matrix(self):
        """
        Return correlation matrix for key numeric columns
        Returns: pd.DataFrame
        """
        numeric_cols = ['Line Amount', 'lead_time','delay_days', 'delay_flag']
        return self.df[numeric_cols].corr()
    
    def plot_bar_chart(self, group_by_col, title, palette="Set2"):
        """
        Reusable bar chart for delay rates grouped by any column.
        Returns: matplotlib.figure.Figure
        """
        grouped_df = self.get_grouped_delay_rate([group_by_col])
        fig, ax = plt.subplots()
        sns.barplot(
            data=grouped_df,
            x=group_by_col,
            y='delay_flag',
            hue=group_by_col,
            palette=palette,
            ax=ax,
            legend=False  # avoid redundant legends
        )
        ax.set_ylabel("Delay Rate")
        ax.set_title(title)
        fig.tight_layout()
        return fig
    
    def plot_monthly_heatmap(self):
        """
        The function `plot_monthly_heatmap` generates a heatmap of delay rates based on year and month data.
        :return: The function `plot_monthly_heatmap` returns a heatmap of delay rates (year vs. month) as a
        matplotlib figure object.
        """
  
        pivot_table = self.get_monthly_delay_trends()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax)
        ax.set_title('Monthly Delay Rate Heatmap')
        fig.tight_layout()
        return fig

    def plot_value_category_comparison(self):
        """
        The function `plot_value_category_comparison` creates a bar plot comparing delay rate and average
        lead time for high value and normal value orders.
        :return: The `plot_value_category_comparison` method is returning a figure object that contains
        two subplots. The first subplot shows the delay rate for high value vs normal value orders, and
        the second subplot shows the average lead time for high value vs normal value orders. Each
        subplot includes bar plots with value labels displayed on top of the bars.
        """
        df = self.df.copy()
        df['Order Value Category'] = df['Line Amount'].apply(
            lambda x: 'High Value (>$5000)' if x > 5000 else 'Normal Value (≤$5000)'
        )

        delay_rate = df.groupby('Order Value Category')['delay_flag'].mean().reset_index()
        lead_time = df.groupby('Order Value Category')['lead_time'].mean().reset_index()

        fig, axs = plt.subplots(2, 1, figsize=(8, 12))
        fig.subplots_adjust(hspace=0.6)  # spacing between subplots

        # --- Plot 1: Delay Rate ---
        sns.barplot(data=delay_rate, x='Order Value Category', y='delay_flag', ax=axs[0], palette='pastel')
        axs[0].set_title("Delay Rate: High Value vs Normal Value Orders", fontsize=17)
        axs[0].set_ylabel("Delay Rate (%)", fontsize=15)
        axs[0].set_xlabel("")
        axs[0].set_ylim(0, 1)
        axs[0].tick_params(labelsize=15)
        # axs[0].bar_label(axs[0].containers[1], fmt="%.1f%%", fontsize=10, padding=3, label_type="center")
        bars = axs[0].patches  # Get the two bars

        for bar in bars:
            height = bar.get_height()
            axs[0].text(
                bar.get_x() + bar.get_width() / 2,
                height - 0.01,  # vertical position 
                f"{height * 100:.1f}%",
                ha='center',
                va='bottom',
                fontsize=14,
                color='black'
            )

        # --- Plot 2: Lead Time ---
        sns.barplot(data=lead_time, x='Order Value Category', y='lead_time', ax=axs[1], palette='muted')
        axs[1].set_title("Average Lead Time: High Value vs Normal Value Orders", fontsize=17)
        axs[1].set_ylabel("Lead Time (Days)", fontsize=15)
        axs[1].set_xlabel("")
        axs[1].tick_params(labelsize=15)
        # axs[1].bar_label(axs[1].containers[0], fmt="%.1f", fontsize=10, padding=3)
        # ✅ Add value labels manually
        bars = axs[1].patches
        for bar in bars:
            height = bar.get_height()
            axs[1].text(
                bar.get_x() + bar.get_width() / 2,
                height - 0.3,  # move slightly below top edge of bar
                f"{height:.1f}",
                ha='center',
                va='bottom',
                fontsize=14,
                color='black'
            )
                

        return fig

    def plot_delay_reasons(self, top_n=10):
        """
        Return a donut chart showing the top delay reasons with a legend instead of labels.
        """
        reason_counts = self.df['ECD Notes'].value_counts()

        # Group smaller reasons into "Other"
        top_reasons = reason_counts[:top_n]
        others = reason_counts[top_n:].sum()
        if others > 0:
            top_reasons['Other'] = others

        # Prepare colors
        colors = plt.cm.tab20.colors  # 20 distinct colors

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            top_reasons,
            labels=None,  # No inline labels
            autopct='%1.1f%%',
            startangle=45,
            wedgeprops=dict(width=0.7),
            colors=colors
        )

        # Add legend outside the plot
        ax.legend(wedges, top_reasons.index, title="Delay Reasons", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

        # Set title and styling
        ax.set_title("Top Delay Reasons (Grouped)", fontsize=14)
        ax.axis('equal')
        fig.tight_layout()
        return fig


    def plot_correlation_matrix(self):
        """
        Return a heatmap figure of the correlation matrix
        Returns: plt.Figure
        """
        corr_matrix = self.get_correlation_matrix()
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title("Correlation Matrix of Key Metrics")
        return fig
    
    def plot_additional_insights(self):
        """
        The `plot_additional_insights` function generates 6 subplots displaying various delay-related
        insights based on the provided data.
        :return: The `plot_additional_insights` method returns a matplotlib figure object that contains 6
        subplots showing additional delay-related insights in a 2x3 grid layout.
        """
        """
        Render 6 subplots in a 2x3 grid showing additional delay-related insights.
        Returns: matplotlib.figure.Figure
        """
        fig, axs = plt.subplots(2, 3, figsize=(10, 6))
        axs = axs.flatten()

        # --- Plot 1: Overall Delay Rate (Pie) ---
        delay_counts = self.df['delay_flag'].value_counts()
        labels = ['Delayed', 'On Time']
        colors = ['#ff9999', '#66b3ff']
        axs[0].pie(delay_counts, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
        axs[0].set_title("Overall Order Delay Status")
        axs[0].axis('equal')

        # --- Plot 2: Delay Rate by Region Zone ---
        region_delay = self.df.groupby('Region Zone')['delay_flag'].mean().sort_values()
        region_delay.plot(kind='bar', ax=axs[1], color=sns.color_palette("Set2"))
        axs[1].set_title("Delay Rate by Region Zone")
        axs[1].set_ylabel("Delay Rate (%)")
        axs[1].tick_params(axis='x', rotation=45)

        # --- Plot 3: URGENT vs Non-URGENT Orders ---
        urgent_delay = self.df.groupby('is_urgent')['delay_flag'].mean().sort_values()
        urgent_delay.index = ['URGENT', 'Non-URGENT'] if 0 in urgent_delay.index else urgent_delay.index
        urgent_delay.plot(kind='bar', ax=axs[2], color=['#2ca02c', '#d62728'])
        axs[2].set_title("Delay Rate: URGENT vs Non-URGENT")
        axs[2].set_ylabel("Delay Rate (%)")
        axs[2].tick_params(axis='x', rotation=0)

        # --- Plot 4: Delay Rate by Branch ---
        branch_delay = self.df.groupby('Top Level Branch')['delay_flag'].mean().sort_values()
        branch_delay.plot(kind='bar', ax=axs[3], color=sns.color_palette("pastel"))
        axs[3].set_title("Delay Rate by Branch")
        axs[3].set_ylabel("Delay Rate (%)")
        axs[3].tick_params(axis='x', rotation=45)


        # --- Plot 5: Delay Rate vs Line Amount ---
        bin_df = self.df.copy()

        # Step 1: Create quantile bins for Line Amount
        bin_df['Line Amount Bin'] = pd.qcut(bin_df['Line Amount'], q=10, duplicates='drop')

        # Step 2: Compute mean delay rate per bin
        grouped = bin_df.groupby('Line Amount Bin')['delay_flag'].mean().reset_index()

        # Step 3: Plot using bin index, but label with actual ranges
        sns.regplot(x=np.arange(len(grouped)), y=grouped['delay_flag'], ax=axs[4], scatter=True)

        # Step 4: Format tick labels using bin intervals
        bin_labels = [f"${int(interval.left):,}–${int(interval.right):,}" for interval in grouped['Line Amount Bin']]
        axs[4].set_xticks(np.arange(len(grouped)))
        axs[4].set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)

        axs[4].set_title("Delay Rate vs Line Amount")
        axs[4].set_xlabel("Line Amount Range")
        axs[4].set_ylabel("Delay Rate")

        # --- Plot 6: Histogram of Delay Days ---
        sns.histplot(self.df['delay_days'].dropna(), bins=30, kde=True, ax=axs[5], color='steelblue')
        axs[5].set_title("Distribution of Delay Days")
        axs[5].set_xlabel("Delay Days")
        axs[5].set_ylabel("Frequency")

        fig.tight_layout()
        return fig
    
    def plot_delay_and_order_trend(self, selected_year, start_month, end_month):
        """
        Plot delay rate and number of orders for selected year and month range.
        Handles both 'All' years and specific years.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Step 1: Filter by month range
        df = self.df.copy()
        df = df[(df['month'] >= start_month) & (df['month'] <= end_month)]

        # Step 2: Filter by year if not 'All'
        if selected_year != "All":
            df = df[df['year'] == selected_year]

        # Step 3: Handle no data
        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data for this selection", ha='center', va='center')
            ax.axis('off')
            return fig

        # Step 4: Prepare complete month range (even if some months missing)
        month_range = pd.DataFrame({'month': list(range(start_month, end_month + 1))})

        # Step 5: Group data
        delay_by_month = df.groupby('month')['delay_flag'].mean().reset_index()
        delay_by_month = month_range.merge(delay_by_month, on='month', how='left')

        orders_by_month = df.groupby('month').size().reset_index(name='order_count')
        orders_by_month = month_range.merge(orders_by_month, on='month', how='left').fillna(0)

        # Step 6: Plot
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'hspace': 0.4})

        title_label = selected_year if selected_year != "All" else "All Years Combined"

        # Plot 1: Delay Rate Line Chart
        sns.lineplot(data=delay_by_month, x='month', y='delay_flag', marker='o', ax=axs[0], color='orange')
        axs[0].set_title(f"Delay Rate - {title_label}")
        axs[0].set_ylabel("Delay Rate")
        axs[0].set_ylim(0, 1)
        axs[0].set_xticks(range(start_month, end_month + 1))
        axs[0].set_xticklabels(range(start_month, end_month + 1))

        # Plot 2: Order Count Bar Chart
        sns.barplot(data=orders_by_month, x='month', y='order_count', ax=axs[1], palette='Blues')
        axs[1].set_title(f"Order Count - {title_label}")
        axs[1].set_ylabel("Number of Orders")
        axs[1].set_xlabel("Month")

        fig.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, cm, labels=None):
        """
        Plot a confusion matrix as a heatmap.
        :param cm: confusion matrix (array-like, shape [n_classes, n_classes])
        :param labels: class labels (optional)
        :return: matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            linewidths=1,
            linecolor='black',
            annot_kws={"size": 10}

        )
        if labels is not None:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted",fontsize=10)
        ax.set_ylabel("Actual",fontsize=10)
        ax.set_title("Confusion Matrix",fontsize=10)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)


        fig.tight_layout()
        return fig
    
    def plot_roc_curve(self, fpr, tpr, auc, figsize=(4, 3)):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}", lw=2)
        ax.plot([0, 1], [0, 1], 'k--', label="Random (AUC=0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig
    
    def plot_feature_importance(self, feat_df, top_n=30, figsize=(6, 6)):
        import matplotlib.pyplot as plt

        feat_df = feat_df.head(top_n)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(feat_df["Feature"][::-1], feat_df.iloc[::-1, 1])  # Plot in reverse order for top-down
        label = "Importance" if "Importance" in feat_df.columns else "Coefficient"
        ax.set_xlabel(label)
        ax.set_title(f"Top {top_n} Feature Importances")
        fig.tight_layout()
        return fig




