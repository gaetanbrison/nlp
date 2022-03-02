## NLP App Theatre Reviews

### Import packages

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import textwrap
from nltk.corpus import stopwords
from pathlib import Path
# Perform standard imports
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy
from annotated_text import annotated_text
import sumy
import spacy_streamlit
from spacy_streamlit import visualize_tokens
from textblob import TextBlob
from gensim.summarization import summarize
import streamlit.components.v1 as components

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

from pathlib import Path
import base64
import time

import text2emotion as te

import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


st.set_page_config(
    page_title="Drama Critiques Playground", layout="wide", page_icon="./images/flask.png"
)




def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

    #images = Image.open('images/binary.png')
    #st.image(images, width=400)

    st.markdown("# Behind the Machine 🔍 🖥")
    st.subheader(
        """
        This is a place where you can get familiar with nlp models  directly from your browser 🧪
        """
    )
    st.markdown("     ")




    selected_indices = []
    master_review = "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles."


    def file_select(folder='./datasets'):
        filelist = os.listdir(folder)
        selectedfile = st.sidebar.selectbox('', filelist)
        return os.path.join(folder, selectedfile)





    df = pd.read_csv("datasets/Corpus_I.csv")


    index_review = 0

    st.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/gaetanbrison/nlp) <small> NLP 4 Critics 1.0.0 | November 2021</small>""".format(
            img_to_bytes("./images/github.png")
        ),
        unsafe_allow_html=True,
    )


    st.sidebar.header("Dashboard")
    st.sidebar.markdown("---")

    st.sidebar.header("Select Project Step")
    nlp_steps = st.sidebar.selectbox('', ['00 - Show  Dataset', '01 - Basic Information',
                                          '02 - Tokenization', '03 - Lemmatization','04 - Name Entity Recognition',
                                          '05 - Part of Speech','06 - Sentiment Analysis',
                                          '07 - Text Summarization'])

    index_review = st.sidebar.number_input("Select a Review by entering index number:", min_value=0, max_value=20000, value=0,
                                   step=1)
    st.markdown("---")
    st.write(f"                                          ")
    if nlp_steps == "00 - Show  Dataset":

        st.header("00 - Show  Dataset")
        if master_review == "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles.":

            head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

            if head == 'Head':
                st.dataframe(df.head(1000))
                #

            else:
                st.dataframe(df.tail(1000))

            st.sidebar.text_area("The review you selected:", value=df['Review'][index_review], height=600)
            st.write(f"                                          ")
            st.write(f"                                          ")


        else:
            num = st.number_input('No. of Rows', 5, 10)
            head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

            if head == 'Head':
                st.dataframe(df.head(num))
                #

            else:
                st.dataframe(df.tail(num))

            st.write(f"                                          ")
            st.write(f"                                          ")


            snippet = f"""
    
            >>> import pandas as pd
            >>> import numpy as  as np
    
            >>> df.head(5)
            #Or
            >>> df.tail(5)
    
            """
            code_header_placeholder = st.empty()
            snippet_placeholder = st.empty()
            code_header_placeholder.subheader(f"**Code for the step: 00 - Show  Dataset**")
            snippet_placeholder.code(snippet)
        st.markdown("---")

    st.write(f"                                          ")
    st.write(f"                                          ")
    if nlp_steps == "01 - Basic Information":
        st.sidebar.text_area("The review you selected:", value=df['Review'][index_review], height=600)
        st.header("01 - Basic Information")
        st.write("Let’s have a look at the main information of this review!")
        if master_review == "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles.":
            st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
            st.text('* Review number of words')
            st.write(len(df["Review"][index_review].split(" ")))
            st.text('* Review number of characters')
            st.write(len(df["Review"][index_review]))

        else:
            st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
            st.text('* Review number of words')
            st.write(len(master_review.split(" ")))
            st.text('* Review number of characters')
            st.write(len(master_review))


        snippet = f"""
    
        >>> import pandas as pd
        >>> import numpy as  as np
    
        # Review number of words
        >>> len(df["Review"][0].split(" "))
    
        # &
        
        # Review number of characters
        >>> len(df["Review"][0])
    
    
        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.subheader(f"**Code for the step: 01 - Basic Information**")
        snippet_placeholder.code(snippet)
        st.markdown("---")

    if nlp_steps == "02 - Tokenization":
        st.header("02 - Tokenization")
        st.sidebar.text_area("The review you selected:", value=df['Review'][index_review], height=600)
        st.write("Tokenization is the act of breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols and other elements called tokens ")
        if master_review == "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles.":
            st.markdown("This is a view of the text split in tokens for the computer to better understand: ")
            doc = nlp(df["Review"][index_review])
            text = df["Review"][index_review].replace(" "," | ")
            st.write(text)

        else:
            st.markdown("This is a view of the text split in tokens for the computer to better understand: ")
            doc = nlp(master_review)
            text = master_review.replace(" "," | ")
            st.write(text)

        snippet = f"""
    
        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
    
        >>> doc = nlp(df["Review"][0])
        >>> text = df["Review"][0].replace(" "," | ")
        >>> print(text)
    
        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.subheader(f"**Code for the step: 02 - Tokenization**")
        snippet_placeholder.code(snippet)
        st.markdown("---")


    st.write(f"                                          ")
    if nlp_steps == "03 - Lemmatization":
        st.header("03 - Lemmatization")
        st.sidebar.text_area("The review you selected:", value=df['Review'][index_review], height=600)
        st.write("""Lemmatization is a linguistic term that means grouping together words with the same root or lemma but with
    different inflections or derivatives of meaning so they can be analyzed as one item. The aim is to take away
    inflectional suffixes and prefixes to bring out the word’s dictionary form. For example, to lemmatize the words
    “cats,” “cat’s,” and “cats’” means taking away the suffixes “s,” “’s,” and “s’” to bring out the root word “cat.”""")
        if master_review == "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles.":
            st.markdown(
                "Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")

            doc = nlp(df["Review"][index_review])
            list_text = []
            list_pos = []
            list_lemma = []
            list_lemma_ = []
            for token in doc:
                list_text.append(token.text)
                list_pos.append(token.pos_)
                list_lemma.append(token.lemma)
                list_lemma_.append(token.lemma_)
            df_lemmatization = pd.DataFrame(
                {'Text': list_text, 'Position': list_pos, 'Unique Code': list_lemma, 'Lemma': list_lemma_, })
            st.dataframe(df_lemmatization,height=1000)

        else:
            st.markdown(
                "Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")

            doc = nlp(master_review)
            list_text = []
            list_pos = []
            list_lemma = []
            list_lemma_ = []
            for token in doc:
                list_text.append(token.text)
                list_pos.append(token.pos_)
                list_lemma.append(token.lemma)
                list_lemma_.append(token.lemma_)
            df_lemmatization = pd.DataFrame(
                {'Text': list_text, 'Position': list_pos, 'Unique Code': list_lemma, 'Lemma': list_lemma_, })
            st.dataframe(df_lemmatization)



        snippet = f"""
    
        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
    
        >>> doc = nlp(df["Review"][0])
        >>> list_text = []
        >>> list_pos = []
        >>> list_lemma = []
        >>> list_lemma_ = []
        >>> for token in doc:
            >>> list_text.append(token.text)
            >>> list_pos.append(token.pos_)
            >>> list_lemma.append(token.lemma)
            >>> list_lemma_.append(token.lemma_)
        >>> df_lemmatization = pd.DataFrame('Text': list_text, 'Position': list_pos, 'Unique Code': list_lemma)
        >>> df_lemmatization
    
        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.subheader(f"**Code for the step: 03 - Lemmatization**")
        snippet_placeholder.code(snippet)
        st.markdown("---")


    if nlp_steps == "04 - Name Entity Recognition":
        st.sidebar.text_area("The review you selected:", value=df['Review'][index_review], height=600)
        st.write(f"                                          ")
        st.header("04 - Name Entity Recognition")
        st.write("""In Natural Language Processing, Named Entity Recognition (NER) is a process where a sentence or a chunk of
    text is parsed through to find entities that can be put under categories like names, organizations, locations, quantities,
    monetary values, percentages, etc. Traditional NER algorithms included only names, places, and organizations.""")
        if master_review == "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles.":
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
            st.markdown("This part assign a tag to each name and entity in a review: ")
            #html = displacy.render(doc, style='ent', jupyter=True)
            #html = html.replace("\n\n", "\n")
            #st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            docx = nlp(df["Review"][0])
            html = displacy.render(docx, style="ent")
            html = html.replace("\n\n", "\n")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

        else:
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
            st.markdown("This part assign a tag to each name and entity in a review: ")
            #html = displacy.render(doc, style='ent', jupyter=True)
            #html = html.replace("\n\n", "\n")
            #st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            docx = nlp(master_review)
            html = displacy.render(docx, style="ent")
            html = html.replace("\n\n", "\n")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

        snippet = f"""
    
        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
        >>> import spacy
        >>> import htbuilder
    
        >>> HTML_WRAPPER = ""<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem"></div>""
    
        >>> docx = nlp(df["Review"][0])
        >>> html = displacy.render(docx, style="ent")
        >>> html = html.replace("\n\n", "\n")
        >>> st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    
    
        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.subheader(f"**Code for the step: 04 - Name Entity Recognition**")
        snippet_placeholder.code(snippet)
        st.markdown("---")






    if nlp_steps == "05 - Part of Speech":
        st.sidebar.text_area("The review you selected:", value=df['Review'][index_review], height=600)
        st.write(f"                                          ")
        st.header("05 - Part of Speech")
        st.write("""Part-of-speech (POS) tagging is a popular Natural Language Processing process which refers to categorizing words
in a text in correspondence with a particular part of speech, depending on the definition of the word and its context.""")
        if master_review == "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles.":
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
            st.markdown("This part assign a tag to each name and entity in a review: ")
            # html = displacy.render(doc, style='ent', jupyter=True)
            # html = html.replace("\n\n", "\n")
            # st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            docx = nlp(df["Review"][0].split(".")[0])
            html = displacy.render(docx, style="dep")
            html = html.replace("\n\n", "\n")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

        else:
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
            st.markdown("This part assign a tag to each name and entity in a review: ")
            # html = displacy.render(doc, style='ent', jupyter=True)
            # html = html.replace("\n\n", "\n")
            # st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            docx = nlp(master_review.split(".")[0])
            html = displacy.render(docx, style="dep")
            html = html.replace("\n\n", "\n")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
        >>> import spacy
        >>> import htbuilder

        >>> HTML_WRAPPER = ""<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem"></div>""

        >>> docx = nlp(df["Review"][0])
        >>> html = displacy.render(docx, style="dep")
        >>> html = html.replace("\n\n", "\n")
        >>> st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.subheader(f"**Code for the step: 05 - Part of Speech**")
        snippet_placeholder.code(snippet)
        st.markdown("---")



# check dupication rate

    #     st.markdown(" Duplication rate is defined as the ratio of  number of duplicates to total records in dataset.")
    #     doc = nlp(df["Review"][index_review])
    #     models = ["en_core_web_sm", "/path/to/model"]
    #     default_text = df["Review"][index_review]
    #     visualizers = ["ner", "textcat"]
    #     spacy_streamlit.visualize(models, default_text, visualizers)
    #
    # else :
    #     st.markdown(" Duplication rate is defined as the ratio of  number of duplicates to total records in dataset.")
    #     doc = nlp(master_review)
    #     models = ["en_core_web_sm", "/path/to/model"]
    #     default_text = master_review
    #     visualizers = ["ner", "textcat"]
    #     spacy_streamlit.visualize(models, default_text, visualizers)
    #
    # snippet = f"""
    #
    # >>> import pandas as pd
    # >>> import numpy as  as np
    # >>> import nltk
    # >>> import spacy
    # >>> import htbuilder
    #
    # >>> doc = nlp(df["Review"][0])
    # >>> models = ["en_core_web_sm", "/path/to/model"]
    # >>> default_text = df["Review"][0]
    # >>> visualizers = ["ner", "textcat"]
    # >>> spacy_streamlit.visualize(models, default_text, visualizers)
    #
    #
    # """
    # code_header_placeholder = st.empty()
    # snippet_placeholder = st.empty()
    # code_header_placeholder.subheader(f"**Code for the step: 06 - Part of Speech**")
    # snippet_placeholder.code(snippet)
    # st.markdown("---")
# Sentiment Analysis
    if nlp_steps == "06 - Sentiment Analysis":
        st.sidebar.text_area("The review you selected:", value=df['Review'][index_review], height=600)
        st.write(f"                                          ")
        st.header("05 - Sentiment Analysis")
        st.write("""Sentiment analysis is the process of detecting positive or negative sentiment in text. It’s often used by businesses
    to detect sentiment in social data, gauge brand reputation, and understand customers.""")
        if master_review == "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles.":
            st.subheader("Analyse Your Text")

            message = st.text_area("Enter Text", df["Review"][index_review],height=250)
            if st.button("Run Sentiment Analysis"):
                blob = TextBlob(message)
                result_sentiment = blob.sentiment
                result_sentiment_2 = te.get_emotion(message)
                nltk.download('vader_lexicon')

                # configure size of heatmap
                sns.set(rc={'figure.figsize': (35, 3)})

                # function to visualize
                def visualize_sentiments(data):
                    sns.heatmap(pd.DataFrame(data).set_index("Sentence").T, center=0, annot=True, cmap="PiYG")

                # text
                sentence = "To inspire and guide entrepreneurs is where I get my joy of contribution"

                # sentiment analysis
                sid = SentimentIntensityAnalyzer()

                # call method
                st.success(sid.polarity_scores(sentence))

                # heatmap

                st.success(result_sentiment)
                st.success(result_sentiment_2)

        else:
            st.subheader("Analyse Your Text")

            message = st.text_area("Enter Text", master_review)
            if st.button("Run Sentiment Analysis"):
                blob = TextBlob(message)
                result_sentiment = blob.sentiment
                result_sentiment_2 = te.get_emotion(message)
                nltk.download('vader_lexicon')

                # configure size of heatmap
                sns.set(rc={'figure.figsize': (35, 3)})

                # function to visualize
                def visualize_sentiments(data):
                    sns.heatmap(pd.DataFrame(data).set_index("Sentence").T, center=0, annot=True, cmap="PiYG")

                # text
                sentence = "To inspire and guide entrepreneurs is where I get my joy of contribution"

                # sentiment analysis
                sid = SentimentIntensityAnalyzer()

                # call method
                st.success(sid.polarity_scores(sentence))

                # heatmap

                st.success(result_sentiment)
                st.success(result_sentiment_2)


        snippet = f"""
    
        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
        >>> import spacy
    
        >>> sid = SentimentIntensityAnalyzer()
        >>> st.success(sid.polarity_scores(sentence))
        >>> blob = TextBlob(message)
        >>> result_sentiment = blob.sentiment
        >>> st.success(result_sentiment)
    
    
        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.subheader(f"**Code for the step: 06 - Sentiment Analysis**")
        snippet_placeholder.code(snippet)
        st.markdown("---")

# Summarization
    if nlp_steps == "07 - Text Summarization":
        st.sidebar.text_area("The review you selected:", value=df['Review'][index_review], height=600)
        st.write(f"                                          ")
        st.header("07 - Text Summarization")
        if master_review == "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles.":
            st.subheader("Summarize Your Text")

            message2 = st.text_area("Review",df["Review"][index_review],height=250)
            summary_options = st.selectbox("Choose Summarizer", ['sumy', 'gensim'])
            if st.button("Summarize"):
                if summary_options == 'sumy':
                    st.text("Using Sumy Summarizer ..")
                    summary_result = sumy_summarizer(message2)
                elif summary_options == 'gensim':
                    st.text("Using Gensim Summarizer ..")
                    summary_result = summarize(message2)
                else:
                    st.warning("Using Default Summarizer")
                    st.text("Using Gensim Summarizer ..")
                    summary_result = summarize(message2)
                st.success(summary_result)

        else:
            st.subheader("Summarize Your Text")

            message2 = st.text_area("Review",master_review)
            summary_options = st.selectbox("Choose Summarizer", ['sumy', 'gensim'])
            if st.button("Summarize"):
                if summary_options == 'sumy':
                    st.text("Using Sumy Summarizer ..")
                    summary_result = sumy_summarizer(message2)
                elif summary_options == 'gensim':
                    st.text("Using Gensim Summarizer ..")
                    summary_result = summarize(message2)
                else:
                    st.warning("Using Default Summarizer")
                    st.text("Using Gensim Summarizer ..")
                    summary_result = summarize(message2)
                st.success(summary_result)




        snippet = f"""
    
        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk
        >>> import sumy
    
        >>> summary_result = summarize(message)
    
        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.subheader(f"**Code for the step: 07 - Text Summarization**")
        snippet_placeholder.code(snippet)
        st.markdown("---")

# # Summarization
#
#     st.write(f"                                          ")
#     st.header("07 - Zoning")
#     if master_review == "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles.":
#         st.subheader("Creation of the zoning")
#         images = Image.open('images/zoning.png')
#         st.image(images, width=None)
#
#
#     else:
#         st.subheader("Creation of the zoning")
#         images = Image.open('images/zoning.png')
#         st.image(images, width=None)
#
#     snippet = f"""
#
#     >>> import pandas as pd
#     >>> import numpy as  as np
#     >>> import nltk
#
#
#     >>> work in progress
#
#     """
#     code_header_placeholder = st.empty()
#     snippet_placeholder = st.empty()
#     code_header_placeholder.subheader(f"**Code for the step: 07 - Zoning**")
#     snippet_placeholder.code(snippet)
#     st.markdown("---")
#
#
#     st.write(f"                                          ")
#     st.header("08 - Map reviews")
#     HtmlFile = open("corpus_I_map_v2.html", 'r', encoding='utf-8')
#     source_code = HtmlFile.read()
#     print(source_code)
#     components.html(source_code, height = 600)
#     st.markdown("---")
#



if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### ** 👨🏼‍💻 App Contributors: **")
st.image(['images/mylene.png','images/gaetan.png'], width=100,caption=["Mylène","Gaëtan"])

st.markdown(f"####  Link to Project Website [here]({'https://dramacritiques.com/fr/accueil/'}) 🚀 ")



def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        " Made by ",
        link("https://odhn.ens.psl.eu/en/newsroom/dans-les-coulisses-des-humanites-numeriques", "Mylène & Gaëtan"),
        " 👩🏼‍💻 👨🏼‍💻"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()



