# The `IntroductionSection` class in the `introduction.py` file of a Streamlit application displays an
# introduction to the dataset including a preview of the data, column descriptions, feature
# engineering details, and features used for model selection.
# src/introduction.py

import streamlit as st

class IntroductionSection:
    @staticmethod
    def show(df):
        st.subheader("📖 Introduction: Understanding the Data")
        st.markdown("Below is a preview of the sales order dataset and a glossary for each column.")

        st.markdown("#### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("#### Column Descriptions")
        st.markdown(
            """
* **Top Level Branch**: Sales branch managing the order (Branch 1, Branch 2, Branch 3, Branch 4)
* **Top Level Sold To**: Client account (customer identifier; many repeated customers)
* **Zone**: Geographic segment (West, East, North, South)
* **Region Zone**: Higher-level geographic segment (Zone A, Zone B, Zone C, Zone D)
* **Top Level Order**: Order identifier (some orders are repeated)
* **Top Level Line**: Order line identifier (can repeat)
* **Last Next Status**: Status of the order (Delivered, On Hold, Shipped, Processing)
* **Shipment Number(s)**: Shipment identifier (can repeat)
* **TL SO Alert**: Alerts or flags (e.g., URGENT)
* **ECD**: Actual Completion Date (when the order was fulfilled)
* **ECD Notes**: Notes on delay reason (e.g., 'demand surge', 'customs backlog', 'holiday backlog', etc.)
* **Line Creation**: When the order was created
* **Top Level Type**: Product information/type
* **Schedule Pick Date**: Planned pickup date
* **Promised Delivery Date**: Promised completion date to the customer
* **Top Level Item**: Product information/identifier
* **Supply Item Description**: Product description (same for all rows)
* **SRP1**: Unit Price
* **SC Rep**: Sales contact/representative
* **Line Amount**: Total amount in dollars
            """
        )

        st.markdown("#### Feature Engineering: Additional Columns Used in the Model")
        st.markdown(
            """
* **delay_days**: Number of days the order was delayed (actual - promised date)
* **delay_flag**: Whether the order was delayed (1 = delayed, 0 = on time)
* **lead_time**: Days from order creation to actual completion (ECD - Line Creation)
* **pickup_lead**: Days from order creation to scheduled pickup
* **month**: Month of promised delivery date (for seasonality trends)
* **year**: Year of promised delivery date
* **week**: Week of promised delivery date
* **is_urgent**: Whether there was an urgent alert (1 = urgent, 0 = not)
* **total_units**: Estimated total units in the order (Line Amount / SRP1)
            """
        )

        st.markdown("#### Features Used for Model Selection")
        st.markdown(
            """
- **Zone, Region Zone, Top Level Branch, Top Level Type, Line Amount, SRP1, Last Next Status, SC Rep, lead_time, is_urgent, total_units**
- These features capture key aspects of geography, customer, product, timing, and urgency to help the model predict delays.
            """
        )
