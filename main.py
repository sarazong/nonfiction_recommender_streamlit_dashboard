import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import pairwise_distances

#***TO re-direct print out from terminal to Streamlit***
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import sys
#***TO re-direct print out from terminal to Streamlit***

# Different sections on the dashboard
header = st.beta_container()
data = st.beta_container()
sample_books = st.beta_container()
recommender1 = st.beta_container()
recommender2 = st.beta_container()

with header:
    st.title("Welcome to my non-fiction book recommender!")
    st.markdown("Some of us read books for mental stimulation, some read for stress reduction, yet others read for knowledge... We read books for different reasons. However, reading a book from end to end is no trivial task. Before we invest our time to start reading a book, it's just natural to ask: is this book worth reading? Or how to choose a book from so many options out there that I would enjoy? This project aims to answer the question by applying natural langague processing on book summaries to explore topics among these books and to build a content-based book recommender.")
    
    
with data:
    #discription of the data used in this project
    st.header("About this dataset:")
    st.markdown("Information for ~3,000 non-fiction books is obtained from Goodreads. Features in this dataset include: (1)title, (2)author, (3)rating, (4)number of rating, (5)number of review, (6)pages, (7)year published, (8)publisher, (9)summary, and (10)topic -- identified from topic modeling.")
    
    #read in the data
    df = pd.read_pickle("data/data_for_streamlit.pkl")
    st.write(df)
    
    #plot number of books per topic identified from summary
    df_plot = pd.DataFrame(df["topic"].value_counts())
    
    plt.style.use("seaborn")
    plt.rcParams["figure.figsize"] = [6, 3]
    fig, ax = plt.subplots()
    bars = ax.bar(df_plot.index, df_plot.topic)
    for bar in range(0, 12, 2):
        bars[bar].set_color("darkmagenta")
    for bar in range(1, 13, 2):
        bars[bar].set_color("plum")
    ax.set_xticks(list(range(12)))
    ax.set_xticklabels(list(df_plot.index), rotation = 60)
    ax.set_ylabel("Number of Books")
    ax.set_title("Number of Books per Topic in the Dataset", weight="bold");
    st.pyplot(fig)
    
    
with sample_books:
    st.header("A few sample books from the dataset:")
    
    #create three columns under sample_books to display book covers
    col_1, col_2, col_3 = st.beta_columns(3)
    with col_1:
        st.image("data/covers/book150.jpg", caption="Full Title: The Coyote's Bicycle: The Untold Story of 7,000 Bicycles and the Rise of a Borderland Empire")
        st.image("data/covers/book1973.jpg", caption="Full Title: Do They Hear You When You Cry ")
    
    with col_2:
        st.image("data/covers/book340.jpg", caption="Full Title: America Alone: The End of the World As We Know It")
        st.image("data/covers/book2913.jpg", caption="Full Title: Endurance: A Year in Space, A Lifetime of Discovery")
        
    with col_3:
        st.image("data/covers/book1651.jpg", caption="Full Title: Grace Will Lead Us Home: The Charleston Church Massacre and the Hard, Inspiring Journey to Forgiveness")
        st.image("data/covers/book3171.jpg", caption="Full Title: The Family Crucible: The Intense Experience of Family Therapy")
    
    
with recommender1:
    st.header("Recommendations for you based on book summary:")
    #read in the data
    df_rec = pd.read_pickle("data/GloVe_embedding_for_recommendation.pkl")
    
    #function for book recommendation based on book summary
    def recommend(title, num_bks = 1):
        title = title.lower()
        ind = pairwise_distances(df_rec.loc[title].values.reshape(1,-1), df_rec, metric="cosine").argsort()[0][1:1+num_bks]
        books = df_rec.index[ind]
        for i in books:
            summary = df.loc[df["title"] == i, "summary"].values[0]
            rating = df.loc[df["title"] == i, "rating"].values[0]
            author = df.loc[df["title"] == i, "author"].values[0]
            print("Title: ", i.title(), "\n")
            print("Rating (scale 0-5): ", rating, "; and Author: ", author, "\n")
            print("Summary: " , summary, "\n")
            
    # re-directing the print out from the recommend function from terminal to streamlit
    @contextmanager
    def st_redirect(src, dst):
        placeholder = st.empty()
        output_func = getattr(placeholder, dst)

        with StringIO() as buffer:
            old_write = src.write

            def new_write(b):
                if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                    buffer.write(b)
                    output_func(buffer.getvalue())
                else:
                    old_write(b)

            try:
                src.write = new_write
                yield
            finally:
                src.write = old_write

    @contextmanager
    def st_stdout(dst):
        with st_redirect(sys.stdout, dst):
            yield

    @contextmanager
    def st_stderr(dst):
        with st_redirect(sys.stderr, dst):
            yield

    # user text input of book title
    title = st.text_input("Please enter the full title of a book you enjoy reading:").strip('"')
    
    # user input on number of books
    num_bks = st.slider("How many book recommendation would you like:", min_value=1, max_value=6, step=1)
    
    # use input to search for books in the dataset
    match = [i.title() for i in df.loc[df["title"].str.contains("^"+title, case=False), "title"]]
    if len(match) == 0:
        st.write("Sorry, no match found in this dataset! Here is a random recommendation for you:")
        title = df.loc[np.random.choice(df.index, size=1), "title"].values[0]
        num_bks = 1
    elif len(match) > 6 and title == "":
        st.write("The following are the first six books from the dataset:", match[:6])
    elif len(match) > 6 and title != "":
        st.write("There are many matches from the dataset and the first six are:", match[:6])
    else:
        st.write("The mataching book from the dataset:", match)

    with st_stdout("info"):
        if title=="":
            print("^ Please enter the title of a book you enjoy reading in the space provide above! ^")
        #elif (len(match) > 6 and title != ""):
        elif (len(match) > 0 and title not in match):
            print("^ Please enter the full title of the book you are interested in! ^")
        else:
            print(recommend(title=title, num_bks=num_bks))


with recommender2:
    st.header("If you are feeling more explorative, here are some random recommendations for you based on topic and rating:")
    
    def recommend2(df, num_bks=1):
        ind = np.random.choice(df.index, size=num_bks, replace=False)
        books = df.loc[ind, :]
        
        for ind in books.index:
            title = books.loc[ind, "title"]
            summary = books.loc[ind, "summary"]
            rating = books.loc[ind, "rating"]
            author = books.loc[ind, "author"]
            print("Title: ", title.title(), "\n")
            print("Rating (scale 0-5): ", rating, "; and Author: ", author, "\n")
            print("Summary: " , summary, "\n")
        
    #create three columns under sample_books to display book covers
    col_1, col_2 = st.beta_columns(2)
    with col_1:
        topics = ["biography", "business", "science", "gender", "religion", "race",
          "health", "world war II", "relationship", "art", "family", "british monarch"]
        topic = st.selectbox("Please select a topic for the book:", options=topics)
    
    with col_2:
        rating = st.number_input("Please enter a number between 0 to 5 for rating (books rated above number entered would be recommended):", min_value=0.0, max_value=5.0, step=0.01, value=4.0)
    
    # user input on number of books
    num_bks2 = st.slider("How many book recommendation would you like:", min_value=1, max_value=6, step=1, key=11)
    
    #select books from specific topic and sort books by rating, descending
    df1 = df[df["topic"] == topic].sort_values("rating")
    match = df1[df1["rating"] >= rating]
    
    if len(match) == 0:
        st.write("Sorry, no match found in this dataset! Here is a random recommendation for you:")
        with st_stdout("info"):
            #randomly recommend a book from the dataset
            print(recommend2(df, num_bks=1))
    elif len(match) < num_bks2:
        st.write(f"Only {len(match)} books from this dataset match the criteria and here they are:")
        with st_stdout("info"):
            print(recommend2(match, num_bks=len(match)))
    else:
        st.write("The recommendations for you are:")
        with st_stdout("info"):
            print(recommend2(match, num_bks=num_bks2))

    
