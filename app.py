import streamlit as st
import requests 
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from geopy.geocoders import OpenCage
import pyodbc as db
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Extract SQL configuration
sql_server = st.secrets["sql"]["SQL_SERVER"]
sql_database_1 = st.secrets["sql"]["SQL_DATABASE_1"]
sql_database_2 = st.secrets["sql"]["SQL_DATABASE_2"]
sql_uid = st.secrets["sql"]["SQL_UID"]
sql_pass = st.secrets["sql"]["SQL_PASS"]

# Access the API key from secrets.toml
opencage_api_key = st.secrets["api"]["OPENCAGE_API_KEY"]

# Extract connection driver
sql_driver = st.secrets["connection"]["driver"]

# Define connection functions for two different databases
def establish_first_db_connection():
    """Establish a connection to the first database."""
    try:
        conn = db.connect(
            f"DRIVER={{{sql_driver}}};"
            f"SERVER={sql_server};"
            f"DATABASE={sql_database_1};"
            f"UID={sql_uid};"
            f"PWD={sql_pass};"
            "Trusted_Connection=no;"
        )
        st.success("Connected")
        return conn
    except db.Error as e:
        st.error(f"Error connecting to the first database: {e}")
        st.stop()

def establish_second_db_connection():
    """Establish a connection to the second database."""
    try:
        conn = db.connect(
            f"DRIVER={{{sql_driver}}};"
            f"SERVER={sql_server};"
            f"DATABASE={sql_database_2};"
            f"UID={sql_uid};"
            f"PWD={sql_pass};"
            "Trusted_Connection=no;"
        )
        st.success("connected")
        return conn
    except db.Error as e:
        st.error(f"Error connecting to the second database: {e}")
        st.stop()


# Page 1: School Nearest Neighbor Finder
def school_nearest_neighbor_page():
    st.title("School Nearest Neighbor Finder")
    st.write("Find the top 5 closest schools by entering a postcode.")
    
    conn = establish_first_db_connection()
    geolocator = OpenCage(api_key=st.secrets["api"]["OPENCAGE_API_KEY"])


    @st.cache_data
    def get_geocode(postcode):
        return geolocator.geocode(postcode)

    @st.cache_data
    def load_data():
        query = """
        SELECT 
            id AS [Record ID],
            property_name AS [Company name], 
            property_post_code AS [Postcode], 
            property_customer_type AS [Customer type - Primary], 
            property_customer_type_re AS [Customer type - RE], 
            property_longitude AS [longitude], 
            property_latitude AS [latitude]
        FROM 
            _hubspot.company;
        """
        df = pd.read_sql_query(query, conn)
        df = df.dropna(subset=['latitude', 'longitude'])
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        primary_saas = df[df['Customer type - Primary'] == 'SaaS'].copy()
        primary_legacy = df[df['Customer type - Primary'] == 'Legacy'].copy()
        re_saas = df[df['Customer type - RE'] == 'SaaS'].copy()
        re_legacy = df[df['Customer type - RE'] == 'Legacy'].copy()
        
        return primary_saas, primary_legacy, re_saas, re_legacy

    primary_saas, primary_legacy, re_saas, re_legacy = load_data()


    postcode = st.text_input("Enter a postcode:", "")
    search_primary_legacy = st.checkbox("Search Primary PSHE", value=True)
    search_jigsaw_re = st.checkbox("Search Jigsaw RE", value=False)
    radius = st.slider("Set Search Radius (in miles)", min_value=1, max_value=50, value=10)
    search_button = st.button("Search")





    if search_button:
        location = get_geocode(postcode)
        selected_datasets = []
        if search_primary_legacy:
            selected_datasets.append(primary_saas)
        if search_jigsaw_re:
            selected_datasets.append(re_saas)

        if location and selected_datasets:
            combined_data = pd.concat(selected_datasets, ignore_index=True)
            nearest_schools = find_nearest_locations(location, combined_data, radius=radius)
            if not nearest_schools.empty:
                st.write(f"Top 5 closest schools within {radius} miles:")
                st.table(nearest_schools)
            else:
                st.error("No schools found within the specified radius.")
        else:
            st.error("Invalid postcode or no datasets selected.")
    
    conn.close()

# Function to find nearest locations (shared function for page 1)
def find_nearest_locations(target_location, data, radius=10):
    if data.empty:
        return pd.DataFrame()
    target_coords = np.radians([[target_location.latitude, target_location.longitude]])
    data_coords = np.radians(data[['latitude', 'longitude']].values.astype(float))
    nbrs = NearestNeighbors(radius=radius * 1609.34, algorithm='ball_tree', metric='haversine')
    nbrs.fit(data_coords)
    distances, indices = nbrs.radius_neighbors(target_coords)
    results = []
    for dist_list, idx_list in zip(distances, indices):
        dist_miles = dist_list * 6371 * 0.621371
        within_radius = [(dist, idx) for dist, idx in zip(dist_miles, idx_list) if dist <= radius]
        top_5_within_radius = sorted(within_radius, key=lambda x: x[0])[:5]
        nearest_data = data.iloc[[idx for _, idx in top_5_within_radius]][['Record ID', 'Company name', 'Customer type - Primary', 'Customer type - RE']].copy()
        nearest_data['Distance (miles)'] = [dist for dist, _ in top_5_within_radius]
        results.append(nearest_data)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

# Page 2: User Search
def user_search_page():
    st.title("Customer Portal Search")
    conn = establish_second_db_connection()

    school_name_input = st.text_input("Enter school name to search (leave empty if not used)")
    postcode_input = st.text_input("Enter postcode to search (leave empty if not used)")

    query = """
    SELECT 
        COALESCE(CONVERT(varchar(10), p1.[dateValue], 103), p1.[varcharValue], p1.[textValue], '') AS 'umbracoMemberLastLogin',
        COALESCE(p5.[varcharValue], p5.[textValue], '') AS 'schoolName',
        COALESCE(p6.[varcharValue], p6.[textValue], '') AS 'postcode'
    FROM 
        [dbo].[umbracoNode] n WITH (NOLOCK)
    JOIN 
        [dbo].[umbracoContent] c WITH (NOLOCK) ON c.[nodeId] = n.[id]
    JOIN 
        [dbo].[cmsMember] m WITH (NOLOCK) ON m.[nodeId] = c.[nodeId]
    JOIN 
        [dbo].[umbracoContentVersion] cv WITH (NOLOCK) ON cv.[nodeId] = m.[nodeId] AND cv.[Current] = 1
    LEFT JOIN 
        [dbo].[cmsPropertyType] pt1 WITH (NOLOCK) ON pt1.[contentTypeId] = c.[ContentTypeId] AND pt1.[Alias] = 'umbracoMemberLastLogin'
    LEFT JOIN 
        [dbo].[umbracoPropertyData] p1 WITH (NOLOCK) ON p1.[versionId] = cv.[id] AND p1.[propertyTypeId] = pt1.[id]
    LEFT JOIN 
        [dbo].[cmsPropertyType] pt5 WITH (NOLOCK) ON pt5.[contentTypeId] = c.[ContentTypeId] AND pt5.[Alias] = 'schoolName'
    LEFT JOIN 
        [dbo].[umbracoPropertyData] p5 WITH (NOLOCK) ON p5.[versionId] = cv.[id] AND p5.[propertyTypeId] = pt5.[id]
    LEFT JOIN 
        [dbo].[cmsPropertyType] pt6 WITH (NOLOCK) ON pt6.[contentTypeId] = c.[ContentTypeId] AND pt6.[Alias] = 'postcode'
    LEFT JOIN 
        [dbo].[umbracoPropertyData] p6 WITH (NOLOCK) ON p6.[versionId] = cv.[id] AND p6.[propertyTypeId] = pt6.[id]
    WHERE  
        n.[level] = 1
    AND 
        n.[trashed] = 0
    """
    
    try:
        df = pd.read_sql(query, conn)
    except db.Error as e:
        st.error(f"Error running query: {e}")
        st.stop()

    if st.button("Submit"):
        filtered_df = df
        if postcode_input:
            filtered_df = filtered_df[filtered_df['postcode'].str.contains(postcode_input, na=False, case=False)]
        if school_name_input:
            filtered_df = filtered_df[filtered_df['schoolName'].str.contains(school_name_input, na=False, case=False)]

        if filtered_df.empty:
            st.warning("No records found for the entered search criteria.")
        else:
            st.success(f"Found {len(filtered_df)} records matching the search criteria.")
            st.dataframe(filtered_df)

    conn.close()



# Page 3: Regional View
def dashboard_page():
    st.title("Regional View")
    st.write("High-level summary of customer distribution and growth insights.")

    # Establish connection to database
    conn = establish_first_db_connection()

    # SQL Query to retrieve data
    query = """
SELECT id, 
       property_name, 
       property_local_authority_reporting_ AS Local_Authority, 
       property_region_dfe_ AS England_Region,
       property_customer_type AS PSHE_Customer_Type, 
       property_customer_type_Re AS RE_Customer_Type,
       property_phaseofeducation_dfe_ AS School_Type,
       property_establishmenttypegroup_dfe_,
       property_longitude as longitude,
       property_latitude as latitude
FROM [_hubspot].[company]
WHERE (property_customer_type IS NOT NULL AND property_customer_type != '') 
   OR (property_customer_type_Re IS NOT NULL AND property_customer_type_Re != '')
  -- AND property_phaseofeducation_dfe_ = 'Primary'
    AND is_deleted = 0


    """
    
    try:
            df = pd.read_sql(query, conn)
    except db.Error as e:
        st.error(f"Error running query: {e}")
        st.stop()



    conn.close()

   # List of regions to exclude
    regions_to_exclude = [
        "Lancashire and West Yorkshire,", 
        "North of England,", 
        "North-West London and South-Central England,", 
        "South-West England,", 
        "West Midlands,",
        "South-East England and South London,",
        ""
    ]

    # Filter out unwanted regions from the DataFrame
    df = df[~df['England_Region'].isin(regions_to_exclude)]


# Sidebar filters
    st.sidebar.subheader("Filters")

    # Multi-select for regions without "All" option
    england_region_options = sorted(df['England_Region'].dropna().unique())
    selected_regions = st.sidebar.multiselect("Select England Regions", england_region_options, default=england_region_options)

    # Other filters for PSHE Customer Type, RE Customer Type, and School Type
    pshe_customer_type_options = ["All"] + list(df['PSHE_Customer_Type'].dropna().unique())
    pshe_customer_type = st.sidebar.selectbox("Filter by PSHE Customer Type", pshe_customer_type_options)

    re_customer_type_options = ["All"] + list(df['RE_Customer_Type'].dropna().unique())
    re_customer_type = st.sidebar.selectbox("Filter by RE Customer Type", re_customer_type_options)

    school_type_options = ["All"] + list(df['School_Type'].dropna().unique())
    school_type = st.sidebar.selectbox("Filter by School Type", school_type_options)

    # Filtered Data based on selected options
    if selected_regions:
        filtered_data = df[df['England_Region'].isin(selected_regions)]
    else:
        # If no region is selected, show all data
        filtered_data = df

    # Further filtering based on other sidebar selections
    filtered_data = filtered_data[
        ((filtered_data['PSHE_Customer_Type'] == pshe_customer_type) | (pshe_customer_type == "All")) &
        ((filtered_data['RE_Customer_Type'] == re_customer_type) | (re_customer_type == "All")) &
        ((filtered_data['School_Type'] == school_type) | (school_type == "All"))
    ]

    # 1. Total Customer Count Metric
    total_customers = len(filtered_data)
    st.metric("Total Customers", total_customers)

    # 2. Customer Count by Region Bar Chart
    customer_count_by_region = filtered_data['England_Region'].value_counts()
    fig_count_region = px.bar(
        customer_count_by_region,
        x=customer_count_by_region.index,
        y=customer_count_by_region.values,
        title="Customer Count by Region",
        labels={"x": "Region", "y": "Customer Count"}
    )
    st.plotly_chart(fig_count_region)

# Assuming 'df' is your DataFrame
    with st.expander("3. Summary Table", expanded=False):
    # Group by region and compute metrics
        region_summary = df.groupby("England_Region").apply(
        lambda group: pd.Series({
            # Total SaaS: Count rows where either PSHE or RE is SaaS (no double counting, excluding blanks)
            "Total_SaaS": ((group["PSHE_Customer_Type"].fillna("") == "SaaS") | 
                           (group["RE_Customer_Type"].fillna("") == "SaaS")).sum(),

            # Total Legacy: Count rows where either PSHE or RE is Legacy (no double counting, excluding blanks)
            "Total_Legacy": ((group["PSHE_Customer_Type"].fillna("") == "Legacy") | 
                             (group["RE_Customer_Type"].fillna("") == "Legacy")).sum(),

            # Separate counts for PSHE SaaS and Legacy (excluding blanks)
            "PSHE_SaaS": (group["PSHE_Customer_Type"].fillna("") == "SaaS").sum(),
            "PSHE_Legacy": (group["PSHE_Customer_Type"].fillna("") == "Legacy").sum(),

            # Separate counts for RE SaaS and Legacy (excluding blanks)
            "RE_SaaS": (group["RE_Customer_Type"].fillna("") == "SaaS").sum(),
            "RE_Legacy": (group["RE_Customer_Type"].fillna("") == "Legacy").sum(),
        })
    ).reset_index()
        
# Additional Table: PSHE and RE SaaS Breakdown
    with st.expander("4. PSHE and RE SaaS Breakdown Table", expanded=False):
    # Group by region and compute metrics for specific criteria
        saas_summary = df.groupby("England_Region").apply(
        lambda group: pd.Series({
            # Customers with "SaaS" in both PSHE and RE
            "Has_Both_PSHE_and_RE": (
                (group["PSHE_Customer_Type"] == "SaaS") & (group["RE_Customer_Type"] == "SaaS")
            ).sum(),

            # Customers with "SaaS" in PSHE and RE is NULL, blank, or Legacy
            "PSHE_SaaS_Only": (
                (group["PSHE_Customer_Type"] == "SaaS") & 
                ((group["RE_Customer_Type"].isna()) | 
                 (group["RE_Customer_Type"] == "") | 
                 (group["RE_Customer_Type"] == "Legacy"))
            ).sum(),

            # Customers with "SaaS" in RE and PSHE is NULL, blank, or Legacy
            "RE_SaaS_Only": (
                (group["RE_Customer_Type"] == "SaaS") & 
                ((group["PSHE_Customer_Type"].isna()) | 
                 (group["PSHE_Customer_Type"] == "") | 
                 (group["PSHE_Customer_Type"] == "Legacy"))
            ).sum(),
        })
    ).reset_index()

    # Append a Totals Row to the SaaS Breakdown Table
    totals = pd.DataFrame([{
        "England_Region": "Total",
        "Has_Both_PSHE_and_RE": saas_summary["Has_Both_PSHE_and_RE"].sum(),
        "PSHE_SaaS_Only": saas_summary["PSHE_SaaS_Only"].sum(),
        "RE_SaaS_Only": saas_summary["RE_SaaS_Only"].sum(),
    }])
    saas_summary = pd.concat([saas_summary, totals], ignore_index=True)

    # Display the SaaS Breakdown Table with compact styling
    st.subheader("PSHE and RE SaaS Breakdown by Region")
    st.dataframe(
        saas_summary.style
            .set_properties(**{
                'text-align': 'center',
                'font-size': '9pt',      # Smaller font size for compactness
                'padding': '0px'         # Remove padding for a compact look
            })
            .set_table_styles([
                {'selector': 'thead th', 'props': [('font-size', '9pt'), ('padding', '0px')]}  # Compact header style
            ]),
        use_container_width=True,
        hide_index=True  # Hide the index column for a cleaner appearance
    )



    # Append a Totals Row to the Summary Table
    totals = pd.DataFrame([{
        "England_Region": "Total",
        "Total_SaaS": region_summary["Total_SaaS"].sum(),
        "Total_Legacy": region_summary["Total_Legacy"].sum(),
        "PSHE_SaaS": region_summary["PSHE_SaaS"].sum(),
        "PSHE_Legacy": region_summary["PSHE_Legacy"].sum(),
        "RE_SaaS": region_summary["RE_SaaS"].sum(),
        "RE_Legacy": region_summary["RE_Legacy"].sum(),
    }])
    region_summary = pd.concat([region_summary, totals], ignore_index=True)

    # Display the Summary Table with compact styling
    st.subheader("Summary Table by Region")
    st.dataframe(
        region_summary.style
            .set_properties(**{
                'text-align': 'center',
                'font-size': '9pt',      # Smaller font size for compactness
                'padding': '0px'         # Remove padding for a compact look
            })
            .set_table_styles([
                {'selector': 'thead th', 'props': [('font-size', '9pt'), ('padding', '0px')]}  # Compact header style
            ]),
        use_container_width=True,
        hide_index=True  # Hide the index column for a cleaner appearance
    )


# Subplots for Total SaaS and Total Legacy by Region with PSHE and RE breakdowns
    fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "RE SaaS vs RE Legacy" , "PSHE SaaS vs PSHE Legacy" 
    )
    )


# PSHE vs RE SaaS Breakdown
    fig.add_trace(
    go.Bar(x=region_summary["England_Region"][:-1], y=region_summary["RE_Legacy"][:-1], name="RE Legacy"),
    row=1, col=1
    )
    fig.add_trace(
    go.Bar(x=region_summary["England_Region"][:-1], y=region_summary["RE_SaaS"][:-1], name="RE SaaS"),
    row=1, col=1
    )

# PSHE vs RE Legacy Breakdown
    fig.add_trace(
    go.Bar(x=region_summary["England_Region"][:-1], y=region_summary["PSHE_Legacy"][:-1], name="PSHE Legacy"),
    row=1, col=2
    )
    fig.add_trace(
    go.Bar(x=region_summary["England_Region"][:-1], y=region_summary["PSHE_SaaS"][:-1], name="PSHE SaaS"),
    row=1, col=2
    )



    fig.update_layout(
    height=500,  # Adjust chart height
    width=900,   # Adjust chart width
    title_text="Customer Type Distribution by Region with PSHE and RE Breakdown",
    showlegend=True
    )


    st.plotly_chart(fig, use_container_width=False)


    # Additional Insights and Visuals
    with st.expander("Additional Insights", expanded=False):
        pshe_type_counts = filtered_data['PSHE_Customer_Type'].value_counts()
        fig_pshe_type = px.pie(pshe_type_counts, names=pshe_type_counts.index, values=pshe_type_counts.values, title="Distribution by PSHE Customer Type")
        st.plotly_chart(fig_pshe_type)

        re_type_counts = filtered_data['RE_Customer_Type'].value_counts()
        fig_re_type = px.pie(re_type_counts, names=re_type_counts.index, values=re_type_counts.values, title="Distribution by RE Customer Type")
        st.plotly_chart(fig_re_type)

    # First Map: Customer Distribution Map
    st.subheader("Customer Distribution Map")
    if not filtered_data.empty:
        fig1 = px.scatter_mapbox(
            filtered_data,
            lat="latitude",
            lon="longitude",
            hover_name="property_name",
            hover_data={
                "Local_Authority": True,
                "PSHE_Customer_Type": True,
                "RE_Customer_Type": True,
                "longitude": False,
                "latitude": False,
            },
            color="PSHE_Customer_Type",
            color_discrete_sequence=px.colors.qualitative.Bold,  # Brighter colors
            zoom=6,
            center={"lat": 52.3555, "lon": -1.1743},  # Center around England
            title="Customer Distribution Map"
        )
        fig1.update_layout(mapbox_style="open-street-map")
        fig1.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig1)
    else:
        st.warning("No data available for the selected filters.")

# Sidebar page selection
page = st.sidebar.selectbox("Select Page", ["School Nearest Neighbor Finder", "User Search", "Regional View"])

# Page routing
if page == "School Nearest Neighbor Finder":
    school_nearest_neighbor_page()
elif page == "User Search":
    user_search_page()
elif page == "Regional View":
    dashboard_page()