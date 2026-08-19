import streamlit as st


st.set_page_config(
    page_title="My Dashboard",
    layout="wide",
)


# Sidebar
st.sidebar.title("My Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Country"]
)


# Main page
if page == "Overview":

    st.title("Overview")

    st.write("Your content goes here.")


elif page == "Country":

    st.title("Country")

    country = st.selectbox(
        "Select a country",
        ["France", "Germany", "United Kingdom"]
    )


    st.title(country)
    st.subheader("Overview 2023")

    st.write(
        "Scientific profile of France and its position "
        "in the global scientific landscape."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Scientific output", "84,231")
        st.metric("Cluster", "Western Europe")

    with col2:
        st.metric("Profile distance", "0.137")
        st.metric("Rank", "12")

    with col3:
        st.write("Plot")
        # st.plotly_chart(fig, use_container_width=True)


    st.divider()


    # a small field-level fingerprint plot (local-global ?)
    # entropy (with quantile)
    # distance local-global (with quantile)

    tab1, tab2, tab3 = st.tabs([
        "Scientific Profile",
        "Profile Stability",
        "Global Position",
    ])

    with tab1:
        st.header("Scientific Profile")
        st.write("fingerprint, national vs international year by year")

    with tab2:
        st.header("Profile Stability")
        st.write("heatmaps + metrics + compare two years")

    with tab3:
        st.header("Global Position")
        st.write("plots + map + distance to cluster profile + alike countries")

    st.write(f"Selected country: {country}")