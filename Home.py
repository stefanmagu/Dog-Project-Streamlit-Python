import pandas as pd
import streamlit as st
from date_out.date import getDate
import matplotlib.pyplot as plt
import plotly.express as px
import wordcloud as wc
import seaborn as sns
import math
import geopandas as gpd

df = getDate()

# labeluri pentru categorii (ca la analiza factoriala) pentru a putea prelucra pt clusterizare (SOLUTIE: alt dataframe...)

st.title("Dog Breed")
st.markdown(
    """
    <style>
    .custom-title {
        color: #F39C12;
        font-size: 40px;
        text-align: center;
        color: red !important;
    }
    /* Change the color of disabled checkboxes */
    div[data-testid="stCheckbox"] label div {
        color: black !important;  /* Change label color */
        font-weight: bold;        /* Make text bold */
    }

    div[data-testid="stCheckbox"] input[disabled] {
        accent-color: #F39C12 !important;  /* Change checkbox color */
        filter: opacity(1) !important;     /* Remove gray effect */
    }    
    </style>
    """,
    unsafe_allow_html=True
)

section = st.sidebar.radio("Navigate to:", ["Search for dog", "Descriptive Analysis","GeoPandas"])

if section == "Search for dog":
    st.header("Search for dog")
    # st.dataframe(df)
    user_select = st.selectbox("Select an option", ["Select an option"] + list(df["Breed"]))

    if user_select != "Select an option":
        row = df[df["Breed"] == user_select]
        st.image(
            row["Image"].values[0],
            width=400,
        )
        c1, c2, c3 = st.columns(3)

        with c1:
            min_height = row["min_height"].values[0]
            max_height = row["max_height"].values[0]
            st.subheader("Height")
            st.markdown(
                f"""
            {int(min_height)}-{int(max_height)} cm    
            """)

        with c2:
            min_weight = row["min_weight"].values[0]
            max_weight = row["max_weight"].values[0]
            st.subheader("Weight")
            st.markdown(
                f"""
                   {int(min_weight)}-{int(max_weight)} kg    
                   """)

        with c3:
            min_expectancy = row["min_expectancy"].values[0]
            max_expectancy = row["max_expectancy"].values[0]
            st.subheader("Life Expectancy")
            st.markdown(
                f"""
                   {int(min_expectancy)}-{int(max_expectancy)} years    
                   """)

        st.subheader("Description")
        st.write(row["description"].values[0])
        st.subheader("Breed Traits & Characteristics")


        tab1, tab2, tab3, tab4 = st.tabs(["Family Life", "Physical", "Social", "Personality"])
        level_labels = {
            1: "Very Low",
            2: "Low",
            3: "Moderate",
            4: "High",
            5: "Very High"
        }
        with tab1:
            st.subheader("Affectionate With Family")
            level = row["Affectionate With Family"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Good With Young Children")
            level = row["Good With Young Children"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Good With Other Dogs")
            level = row["Good With Other Dogs"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

        with tab2:
            st.subheader("Coat Type")
            coattypes = list(df["Coat Type"].unique())
            coattype = row["Coat Type"].values[0]
            c1, c2 = st.columns(2)
            with c1:
                for i in coattypes[0:5]:
                    if i == coattype:
                        st.checkbox(i, value=True, disabled=True, key=f'{i}')
                    else:
                        st.checkbox(i, disabled=True, key=f'{i}')
            with c2:
                for i in coattypes[5:]:
                    if i == coattype:
                        st.checkbox(i, value=True, disabled=True, key=f'{i}')
                    else:
                        st.checkbox(i, disabled=True, key=f'{i}')

            st.subheader("Coat Length")

            coatLengths = list(df["Coat Length"].unique())
            coatLength = row["Coat Length"].values[0]

            c1, c2 = st.columns(2)
            with c1:
                for i in coatLengths[0:2]:
                    if i == coatLength:
                        st.checkbox(i, value=True, disabled=True, key=f'{i}')
                    else:
                        st.checkbox(i, disabled=True, key=f'{i}')
            with c2:
                for i in coatLengths[2:]:
                    if i == coatLength:
                        st.checkbox(i, value=True, disabled=True, key=f'{i}')
                    else:
                        st.checkbox(i, disabled=True, key=f'{i}')

            st.subheader("Shedding Level")
            level = row["Shedding Level"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Coat Grooming Frequency")
            level = row["Coat Grooming Frequency"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Drooling Level")
            level = row["Drooling Level"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

        with tab3:
            st.subheader("Openness To Strangers")
            level = row["Openness To Strangers"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Playfulness Level")
            level = row["Playfulness Level"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Watchdog/Protective Nature")
            level = row["Watchdog/Protective Nature"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Adaptability Level")
            level = row["Adaptability Level"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

        with tab4:
            st.subheader("Trainability Level")
            level = row["Trainability Level"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Energy Level")
            level = row["Energy Level"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Barking Level")
            level = row["Barking Level"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

            st.subheader("Mental Stimulation Needs")
            level = row["Mental Stimulation Needs"].values[0]
            st.progress(int(level) * 20, text=f"Level: {level} - {level_labels[level]}")

        st.header("Word Cloud")
        wc = wc.WordCloud(width=800, height=800, background_color='white').generate(row["description"].values[0])
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig=plt)

elif section == "Descriptive Analysis":
    st.header("Descriptive Analysis")

    st.write(df.describe())

    df_grades = df.iloc[:, :17].set_index("Breed")
    df_grades.drop(columns=["Coat Type", "Coat Length"], inplace=True)

    # mean
    st.subheader("Mean grades")
    st.bar_chart(df_grades.describe().loc["mean"], x_label="Mean Grades", horizontal=True)
    st.html('<p style="color:blue;">Most dog breeds are affectionate with family members, with an average score of 4.5 out of 5</p>')

    c1, c2 = st.columns(2)
    with c1:
        coat_type_fig = px.pie(df, names="Coat Type", title="Coat Type Distribution")
        st.plotly_chart(coat_type_fig)

    with c2:
        coat_length_fig = px.pie(df, names="Coat Length", title="Coat Length Distribution")
        st.plotly_chart(coat_length_fig)
    st.html('<p style="color:blue;">Most dog breeds have a smooth coat type and short coat length</p>')

    # TOP BREEDS
    df_grades["Total"] = df_grades.sum(axis=1)
    st.subheader("Top 5 breeds by total grade")
    st.write(df_grades["Total"].sort_values(ascending=False).head(5))
    st.html('<p style="color:blue;">The breed with the highest score overall is Portuguese Water Dog</p>')

    # BOTTOM BREEDS
    st.subheader("Bottom 5 breeds by total grade")
    st.write(df_grades["Total"].sort_values(ascending=True).head(5))

    # TOP BREEDS BY HEIGHT, WEIGHT, LIFE EXPECTANCY
    st.subheader("Top 5 breeds by height")
    st.write(df.groupby("Breed")[["min_height", "max_height"]].mean().mean(axis=1).sort_values(ascending=False).head(
        5).rename("Average Height"))

    st.subheader("Top 5 breeds by weight")
    st.write(df.groupby("Breed")[["min_weight", "max_weight"]].mean().mean(axis=1).sort_values(ascending=False).head(
        5).rename("Average Weight"))

    st.subheader("Top 5 breeds by life expectancy")
    st.write(
        df.groupby("Breed")[["min_expectancy", "max_expectancy"]].mean().mean(axis=1).sort_values(ascending=False).head(
            5).rename("Average Life Expectancy"))


    # MOST USED WORDS in description
    st.subheader("Most used words in description")
    text = " ".join(df["description"])
    wc = wc.WordCloud(width=800, height=800, background_color='white').generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig=plt)


    # CORRELATION MATRIX
    numerical_cols = df_grades.drop(columns=["Total"]).columns
    st.subheader("Correlation matrix")
    corr_matrix = df_grades[numerical_cols].corr()
    # Vizualizăm matricea de corelație cu un heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix for Breed Traits")
    plt.tight_layout()
    st.pyplot(fig=plt)

    st.html("""
            <p style='color:blue;'>
            The correlation matrix shows that the traits are not strongly correlated with each other. 
            </p>

            <ul style='color:blue;'>
                <li>The highest correlation is between Mental Stimulation Needs and Energy Level, with a value of 0.64, meaning energetic breeds often require more mental challenges.</li>
                <li>Playfulness Level correlates notably with Openness To Strangers (0.46) and Adaptability Level (0.49), indicating playful breeds tend to be friendlier and more adaptable.</li>

            </ul>    
            """)


    # HISTOGRAMS - FREQUENCY DISTRIBUTION
    st.subheader("Frequency distribution of grades")
    n_cols = 2
    n_rows = math.ceil(len(numerical_cols) / n_cols)

    # Create subplots
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    ax = ax.flatten()  # Pentru iterare

    for i, col in enumerate(numerical_cols):
        value_counts = df[col].value_counts().sort_index()

        ax[i].bar(value_counts.index, value_counts.values, color='skyblue', edgecolor='black')
        ax[i].set_title(f'Distribution: {col}')
        ax[i].set_xlabel(col)
        ax[i].set_ylabel('Frequency')
        ax[i].set_xticks([1, 2, 3, 4, 5])
        ax[i].set_xlim(0, 6)
    plt.tight_layout()
    st.pyplot(fig=fig)

elif section == "GeoPandas":
    st.title("Searching dogs around the world")

    # Citire Fisier
    df_world = pd.read_csv("date_in/Dog Breads Around The World.csv")
    df_world = df_world.iloc[:, :2]
    df_world["Origin"] = df_world["Origin"].replace({"USA": "United States of America"})
    df_world["Origin"] = df_world["Origin"].replace({"Alaska USA": "United States of America"})
    df_world["Origin"] = df_world["Origin"].replace({"UK": "United Kingdom"})

    # Prelucrare Dataframe
    world = gpd.read_file("date_in/custom.geo.json")
    world = world[["name", "geometry"]]
    world = world.rename(columns={"name": "Origin"})
    world = world.merge(df_world, on="Origin", how="left")

    # Generare Harta
    world["color"] = world["Name"].apply(lambda x: "#00ffc3" if pd.notna(x) else "#ffffff")

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_title('Dog breeds around the world')
    world.plot(color=world["color"], ax=ax, edgecolor="black")
    legend_labels = {"#00ffc3": "Has Dog Breed", "#ffffff": "No Dog Breed"}
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
               for color in legend_labels.keys()]
    ax.legend(handles, legend_labels.values(), loc="lower left")
    st.pyplot(fig)


    # Search for Dog Breed
    st.subheader("Search for dog")
    search = st.selectbox("Select an option", ["Select an option"] + list(world["Name"].dropna().unique()))
    if search != "Select an option":
        row = world[world["Name"] == search]

        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        ax.set_title(f"{search} - Country of Origin: {row['Origin'].values[0]}")

        # Highlight only the selected country
        world.plot(color="#ffffff", ax=ax, edgecolor="black")  # Default map in white
        row.plot(color="#00ffc3", ax=ax, edgecolor="black")  # Highlight selected country
        ax.set_aspect("auto")
        st.pyplot(fig)

    # Search for Dog Breed by Country
    st.subheader("Search for dog by country")
    country_search = st.selectbox("Select a country", ["Select a country"] + list(world["Origin"].dropna().unique()))
    if country_search != "Select a country":
        row = world[world["Origin"] == country_search]

        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        ax.set_title(f"{country_search} - Dog Breeds")

        # Preserve the original map bounds
        world.plot(color="#ffffff", ax=ax, edgecolor="black")  # Default map in white
        row.plot(color="#00ffc3", ax=ax, edgecolor="black")  # Highlight selected country
        ax.set_aspect("auto")

        st.pyplot(fig)

        # Display list of dog breeds in the selected country
        breeds = row["Name"].dropna().tolist()
        if breeds:
            st.write(f"Dog breeds from {country_search}:")
            st.table(breeds)
        else:
            st.write(f"No known dog breeds from {country_search}.")





