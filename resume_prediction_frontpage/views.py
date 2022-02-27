import base64

from IPython.display import display
from django.shortcuts import render
#from resume_prediction_frontpage import models
from django.http import HttpResponse
from .models import File, Contact
#from resume_prediction_frontpage.forms import FileForm
from django.views.generic.edit import CreateView
from django.urls import reverse_lazy
from django.core.files.storage import FileSystemStorage
from datetime import datetime
from django.contrib import messages
from io import BytesIO

# Create your views here.



def home(request):
    return render(request, "base.html")


def index(request):
    context={'a':'Welcome to the application!'}
    return render(request,'index.html',context)
    # return HttpResponse({'abhishek':1})



class UploadView(CreateView):
    model = File
    fields = ['upload_file', ]
    success_url = reverse_lazy('fileupload')
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['documents'] = File.objects.all()
        context['prediction'] = 'yes'
        print('value in context')
        print(context['documents'])
        return context





def upload_resume(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
        scoreval = "uploaded"
        context['scoreval'] = scoreval
    return render(request, 'upload_resume.html', context)

'''
def showfile(request):
    lastfile = File1.objects.last()

    filepath = lastfile.filepath

    filename = lastfile.name

    form = FileForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    context = {'filepath': filepath,
               'form': form,
               'filename': filename
               }

    return render(request, 'get_resume/files.html', context)

'''
#from .serializers import FileSerializer
def upload(request):
    context = {}
    try:
        if request.method == 'POST':
            uploaded_file = request.FILES['document']
            fs = FileSystemStorage()
            name = fs.save(uploaded_file.name, uploaded_file)
            context['url'] = fs.url(name)
            scoreval = "uploaded"
            context['scoreval'] = scoreval
            print(fs)
            print('link',fs.url(name))
            
        else:
            scoreval = "Not Uploaded"
            context['scoreval'] = scoreval
        return render(request, 'upload.html', context)
    except KeyError:
        return Response("file_serializer.errors", status=status.HTTP_404_NOT_FOUND)

def multiple_uploads(request):
    dir = './media'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES.getlist('document')
        fs = FileSystemStorage()
        allfile =[]
        countfile = 0
        for f in uploaded_file:
            name = fs.save(f.name, f)
            context['url'] = fs.url(name)
            allfile.append(f.name)
            countfile = countfile+1
        context['allfile'] = allfile
        context['countfile'] = countfile


    return render(request, 'multiple_uploads.html', context)

def delete(request):
    dir = './media'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    context = {}

    return render(request, 'delete.html')

def solutions(request):
    return render(request, 'solutions.html')



def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        desc = request.POST.get('desc')
        contact = Contact(name=name, email=email, phone=phone, desc=desc, date = datetime.today())
        contact.save()
        messages.success(request, 'Your message has been sent!')
    return render(request, 'contact.html')

def aboutus(request):
    return render(request, 'aboutus.html')

import pip

def install(package):
    pip.main(['install', package,'--user'])

# Example
if __name__ == '__main__':
    install('scikit-learn')
    install('pandas')

import glob
import os
import joblib
import pdfplumber
import pandas as pd
import numpy as np
import logging
# to ignore logs from pdfplumber
logging.getLogger("pdfminer").setLevel(logging.WARNING)
#python -m ensurepip - to get pip back

from os import listdir
from os.path import isfile, join


#NLTK Packages
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

import re
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words('english')
# Importing spacy
#import spacy

# Loading model
#!python -m spacy download en
#nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
#nltk.download('all')
#nltk.download('')

#wordcloud-1.8.1-cp39-cp39-win_amd64



import time


import matplotlib.pyplot as plt
from io import StringIO
#%matplotlib inline

import plotly as py
import cufflinks as cf

from plotly.offline import iplot


from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
#py.offline.init_notebook_mode(connected=True)
#cf.go_offline()


from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")
#from yellowbrick.text import TSNEVisualizer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.metrics.pairwise import cosine_similarity

from termcolor import colored

reloadModel = joblib.load('./models/LR_TransformedTargetRegressor_WithGridFeatures_WithSimilarity_6_52.pkl')
#reloadModel = joblib.load('A:/UOH_Diploma_AI_ML/projects/ml_apps/resume_prediction_ml_apps/models/knn_model.pkl')
#reloadvectorizer = joblib.load('./models/vectorizer.pkl')

# Defining a method to read text from pdf file using pdfplumber

def read_pdf_data(latest_data_path, input_file_name):
    '''
    input - path of the directory in the following format as e.g:'A:/UOH_Diploma_AI_ML/projects/dataset/trainResumes'

    output - text of the pdf file as tmp_text object

    '''

    tmp_text = []
    with pdfplumber.open(latest_data_path + "/" + input_file_name) as pdf:
        totalpages = len(pdf.pages)
        for pageno in range(totalpages):
            page_text = pdf.pages[pageno]
            try:
                txt = page_text.extract_text()
            except:
                pass
            else:
                tmp_text.append(txt)
    return tmp_text


def pdf_text_to_df(input_file_list, latest_data_path):
    '''
    input1 - list of input files for which text etraction required from pdf files
    input1 - path of the directory in the following format as e.g:'A:/UOH_Diploma_AI_ML/projects/dataset/trainResumes'

    output - text of the pdf file as pandas dataframe

    '''

    resume_text_df = pd.DataFrame()
    for ix in range(len(input_file_list)):
        pdf_data = read_pdf_data(latest_data_path, input_file_list[ix])

        # Available resume format has following structure and properties:
        #         1. Line by line data is available to read as the text is delimited by new line character i.e by '\n'
        #         2. Also the sentences are separated by one special character which is '●'.
        #         3. First line usually contain Name of the candidate, 2nd line usually is role or what the candidate aspire to be
        #
        # So as per the analysis , names and special characters are not significant thus following are preprocessing steps
        # need to apply
        # complete_text = ' '.join(pdf_data).replace('\n',' ').replace('●','')
        complete_text = ' '.join(' '.join(pdf_data).split('\n')[1:])

        # Also

        resume_text_df.loc[ix, 'CandidateID'] = input_file_list[ix].replace('.pdf', '')
        resume_text_df.loc[ix, 'resume_text'] = complete_text
    return resume_text_df

def get_total_text(text_series):
    total_resume_text = ''
    for txt in text_series:
        txt = str(txt)
        token = txt.split()
        total_resume_text += " ".join(token)+" "

    return total_resume_text


def get_wordcloud(total_resume_text):
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10).generate(total_resume_text)

    return wordcloud


def get_lowercase(text_list):
    '''
    input - all the text in a list object

    output - lower case text as list object

    '''
    lowertext_list = []
    for i in text_list:
        i = str(i)
        split_text = i.split()
        lowertext = [x.lower() for x in split_text]
        lowertext1 = " ".join(lowertext)
        lowertext_list.append(lowertext1)
    return lowertext_list


def remove_nltk_stopwords(text_list):
    '''
    input - all the text in a list object

    output - lower case text as list object

    '''
    text_without_stopwords = []
    for txt in text_list:
        content_tokens = word_tokenize(txt)
        nostopwords = [x for x in content_tokens if x not in stopwords and len(x) > 2]
        contentnostopwords = " ".join(nostopwords)
        text_without_stopwords.append(contentnostopwords)
    return text_without_stopwords


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = ' '.join([stemmer.stem(t) for t in filtered_tokens])
    return stems


def tokenize_and_lemma_NLTK(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    lemma = ' '.join([lem.lemmatize(t) for t in filtered_tokens])

    return lemma


def tokenize_and_lemma_SPACY(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    lemma = ' '.join([t.lemma_ for t in filtered_tokens])

    return lemma


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# function to provide list of all files from the given path of the directory
def list_all_files(input_file_path):
    '''
    input - path of the directory where files are stored in the following format
    as e.g:'A:/UOH_Diploma_AI_ML/projects/dataset/trainResumes'

    output - list of names of the pdf file in the directory path as 'file_list_tmp' --> list object

    '''

    file_list_tmp = []
    for f in listdir(input_file_path):
        if isfile(join(input_file_path, f)):
            file_list_tmp.append(f)
    return file_list_tmp

def get_graph():

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

# plot the WordCloud image
def plot_wordcloud(text):
    wordcloud = get_wordcloud(text)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    plt.figure(figsize=(10, 5), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    #plt.show()

    #fig = plt.figure()
    #imgdata = StringIO()
    #fig.savefig(imgdata, format='svg')
    #imgdata.seek(0)

    #data = imgdata.getvalue()
    data = get_graph()
    return data


def compare_jd_vs_resume(idx,data,tokenized_jobtext):
    resume_id = data.loc[idx,'CandidateID']
    complete_text = get_total_text(data[data['CandidateID']==resume_id]['clean_text'])
    matching_words = list(set(complete_text.split())&set(tokenized_jobtext[0].split()))
    matching_word_length = len(matching_words)
    matching_word_length_percentage = round(100*len(matching_words)/len(tokenized_jobtext[0].split()),2)
    print(colored('Following are the {} similar words list which are matching with the job description with {}% are : \n {}\n'.format(matching_word_length,matching_word_length_percentage,matching_words),'red', attrs=['bold']))

    input_cloud = plot_wordcloud(complete_text)

    return matching_words,matching_word_length,matching_word_length_percentage,input_cloud





def clean_jd():
    # processing job description
    # Listing all the pdf file for the job description data
    jobdes_data_path = './jobdescription'
    jobdes_pdffiles = list_all_files(jobdes_data_path)
    print('There is only {} job description and that is {}'.format(len(jobdes_pdffiles), jobdes_pdffiles[0:5]))

    # Read the text of the training resumes and storing as a pandas dataframe
    start_time1 = time.time()
    job_description_df = pdf_text_to_df(jobdes_pdffiles, jobdes_data_path)
    print("--- %s seconds ---" % (time.time() - start_time1))
    job_description_df = job_description_df.rename(columns={'resume_text': 'job_description_text'})
    job_description_df.head()

    # Pre-processing Job role Data : 1 - lower case conversion of content

    lower_jobtextlist = get_lowercase(job_description_df['job_description_text'])
    job_description_df['job_description_text'] = lower_jobtextlist

    # Pre-processing Data : 2 for job description
    jobtextlist_without_stopwords = remove_nltk_stopwords(job_description_df['job_description_text'])
    job_description_df['job_description_text'] = jobtextlist_without_stopwords

    tokenized_jobtext = [tokenize_and_lemma_NLTK(text) for text in
                         job_description_df['job_description_text'].tolist()]

    job_description_df['job_description_text'] = tokenized_jobtext
    return job_description_df,tokenized_jobtext

def get_cosine_similarity(jddf,inputdf):
    sk_cosine_sim = []
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    for i in inputdf['clean_text']:
        text_list = [i, jddf['job_description_text'][0]]
        count_matrix = cv.fit_transform(text_list)
        sim_val = cosine_similarity(count_matrix)[0][1] * 100
        sk_cosine_sim.append(round(sim_val, 2))
    inputdf['sklearn_cosine_similarity'] = sk_cosine_sim

    return inputdf


# can use spacy for lemmatization - later on


def predict(request):
    import time
    start_time1 = time.time()
    print(request)
    if request.method == 'GET':
        latest_data_path = './media'
        list_of_files = glob.glob(
            './media/*.pdf')  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getmtime)
        lastest_pdf = str(latest_file).split("\\")[1]

        input_file_list = []
        input_file_list.append(lastest_pdf)

        # Read the text of the training resumes and storing as a pandas dataframe
        import time
        input_resume_df = pdf_text_to_df(input_file_list, latest_data_path)

        # Lets view few text and try to understand
        #resume_text = input_resume_df.loc[0, 'resume_text']

        # Pre-processing Data : 1 - lower case conversion of content


        lower_textlist = get_lowercase(input_resume_df['resume_text'])
        input_resume_df['resume_text'] = lower_textlist

        # Pre-processing Data : 2 - Remove stop words and also words with length of character less than or equal to 2 for e.g of,it, I etc.

        textlist_without_stopwords = remove_nltk_stopwords(input_resume_df['resume_text'])
        input_resume_df['resume_text'] = textlist_without_stopwords

        # tokenize and lemmatize
        #% time
        tokenized_text = [tokenize_and_lemma_NLTK(text) for text in input_resume_df['resume_text'].tolist()]

        input_resume_df['clean_text'] = tokenized_text
        input_resume_df = input_resume_df[['CandidateID', 'clean_text']]
        #saving the input data
        input_resume_df.to_csv('./input_data/input_resume_df.csv',index=False)

        job_description_df,tokenized_jobtext = clean_jd()

        '''
        sk_cosine_sim = []
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer()
        for i in input_resume_df['clean_text']:
            text_list = [i, job_description_df['job_description_text'][0]]
            count_matrix = cv.fit_transform(text_list)
            sim_val = cosine_similarity(count_matrix)[0][1] * 100
            sk_cosine_sim.append(round(sim_val, 2))
        input_resume_df['sklearn_cosine_similarity'] = sk_cosine_sim
        '''

        input_resume_df = get_cosine_similarity(job_description_df,input_resume_df)
        # check the fitted vocabulary on test data sample
        #input_vector = reloadvectorizer.transform(input_resume_df['clean_text'])
        #bowfeature = reloadvectorizer.get_feature_names()
        # document term matrix and vocabulary together for test data
        #input_vector_df = pd.DataFrame(input_vector.toarray(), columns=bowfeature, index=input_resume_df.index)
        #input_vector_df['similarity_score'] = input_resume_df['sklearn_cosine_similarity']
        numeric_features = ['sklearn_cosine_similarity']
        # numeric_features = []
        categorical_features = 'clean_text'
        featurelist = numeric_features
        featurelist.append(categorical_features)
        finalfeatures = featurelist
        input_vector_df = input_resume_df[featurelist]
        input_vector_df = input_resume_df[finalfeatures]

        prediction = reloadModel.predict(input_vector_df)
        scoreval = round(prediction[0][0],2)
        scoredf = pd.DataFrame()
        scoredf.loc[0,'score'] = scoreval
        scoredf.to_csv('./score/scoredf.csv',index=False)

        '''
        # index of the resume for which further analysis and comparison is required
        idx = 0  # please change the index as per your requirement.
        compare_df = input_resume_df  # X_train # we want to compare the results from test data thus using X_test, it can be changed to X_train
        resume_id = compare_df.loc[idx, 'CandidateID']

        print(colored('Comparison Between Candidate {} and Job Description'.format(resume_id), 'green', attrs=['bold']))
        print(colored('Word Cloud For Candidate {}'.format(resume_id), 'blue', attrs=['bold']))
        compare_jd_vs_resume(idx, compare_df,tokenized_jobtext)
        print(colored('Word Cloud For Job Description', 'blue', attrs=['bold']))
        graph = plot_wordcloud(tokenized_jobtext[0])
        '''
        total_time = ' : ' + str(round((time.time() - start_time1), 2)) + ' seconds'
        context = {'scoreval': scoreval,'total_time':total_time}

    return render(request, 'predict.html', context)

from IPython.display import HTML

def predict_all(request):
    import time
    start_time1 = time.time()
    print(request)
    if request.method == 'GET':
        latest_data_path = 'A:/UOH_Diploma_AI_ML/projects/ml_apps/resume_prediction_ml_apps/media'
        list_of_files = glob.glob(
            'A:/UOH_Diploma_AI_ML/projects/ml_apps/resume_prediction_ml_apps/media/*.pdf')  # * means all if need specific format then *.csv

        input_file_list = []
        for f in list_of_files:
            lastest_pdf = str(f).split("\\")[1]
            input_file_list.append(lastest_pdf)

        # Read the text of the training resumes and storing as a pandas dataframe
        import time
        input_resume_df = pdf_text_to_df(input_file_list, latest_data_path)

        # Lets view few text and try to understand
        #resume_text = input_resume_df.loc[0, 'resume_text']

        # Pre-processing Data : 1 - lower case conversion of content


        lower_textlist = get_lowercase(input_resume_df['resume_text'])
        input_resume_df['resume_text'] = lower_textlist

        # Pre-processing Data : 2 - Remove stop words and also words with length of character less than or equal to 2 for e.g of,it, I etc.

        textlist_without_stopwords = remove_nltk_stopwords(input_resume_df['resume_text'])
        input_resume_df['resume_text'] = textlist_without_stopwords

        # tokenize and lemmatize
        #% time
        tokenized_text = [tokenize_and_lemma_NLTK(text) for text in input_resume_df['resume_text'].tolist()]

        input_resume_df['clean_text'] = tokenized_text
        input_resume_df = input_resume_df[['CandidateID', 'clean_text']]
        #saving the input data
        input_resume_df.to_csv('A:/UOH_Diploma_AI_ML/projects/ml_apps/resume_prediction_ml_apps/input_data/input_resume_df.csv',index=False)

        job_description_df,tokenized_jobtext = clean_jd()

        '''
        sk_cosine_sim = []
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer()
        for i in input_resume_df['clean_text']:
            text_list = [i, job_description_df['job_description_text'][0]]
            count_matrix = cv.fit_transform(text_list)
            sim_val = cosine_similarity(count_matrix)[0][1] * 100
            sk_cosine_sim.append(round(sim_val, 2))
        input_resume_df['sklearn_cosine_similarity'] = sk_cosine_sim
        '''

        input_resume_df = get_cosine_similarity(job_description_df,input_resume_df)
        # check the fitted vocabulary on test data sample
        #input_vector = reloadvectorizer.transform(input_resume_df['clean_text'])
        #bowfeature = reloadvectorizer.get_feature_names()
        # document term matrix and vocabulary together for test data
        #input_vector_df = pd.DataFrame(input_vector.toarray(), columns=bowfeature, index=input_resume_df.index)
        #input_vector_df['similarity_score'] = input_resume_df['sklearn_cosine_similarity']
        numeric_features = ['sklearn_cosine_similarity']
        # numeric_features = []
        categorical_features = 'clean_text'
        featurelist = numeric_features
        featurelist.append(categorical_features)
        finalfeatures = featurelist
        input_vector_df = input_resume_df[featurelist]
        input_vector_df = input_resume_df[finalfeatures]

        predictions = reloadModel.predict(input_vector_df.loc[0:]).tolist()
        predlist = []
        for i in range(0, input_vector_df.shape[0]):
            predlist.append(str(round(predictions[i][0], 2)) + '%')

        input_vector_df.loc[:, 'Prediction_Score'] = predlist

        predictiondf = pd.DataFrame()
        predictiondf = input_vector_df.copy()
        predictiondf['CandidateID'] = list(input_resume_df['CandidateID'])
        predictiondf = predictiondf.rename(columns={'sklearn_cosine_similarity': 'Similarity_Score'})
        predictiondf = predictiondf[['CandidateID', 'Similarity_Score', 'Prediction_Score']]
        predictiondf = predictiondf.sort_values(by='Prediction_Score',ascending=False)
        predictiondf = predictiondf.reset_index(drop=True)
        predictiondf.index = np.arange(1, len(predictiondf) + 1)
        predictiondf_html = predictiondf.to_html()
        text_file = open("./prediction_data.html", "w")
        text_file.write(predictiondf_html)
        text_file.close()

        #scoreval = round(prediction[0][0],2)
        #scoredf = pd.DataFrame()
        #scoredf.loc[0,'score'] = scoreval
        #scoredf.to_csv('A:/UOH_Diploma_AI_ML/projects/ml_apps/resume_prediction_ml_apps/score/scoredf.csv',index=False)

        '''
        # index of the resume for which further analysis and comparison is required
        idx = 0  # please change the index as per your requirement.
        compare_df = input_resume_df  # X_train # we want to compare the results from test data thus using X_test, it can be changed to X_train
        resume_id = compare_df.loc[idx, 'CandidateID']

        print(colored('Comparison Between Candidate {} and Job Description'.format(resume_id), 'green', attrs=['bold']))
        print(colored('Word Cloud For Candidate {}'.format(resume_id), 'blue', attrs=['bold']))
        compare_jd_vs_resume(idx, compare_df,tokenized_jobtext)
        print(colored('Word Cloud For Job Description', 'blue', attrs=['bold']))
        graph = plot_wordcloud(tokenized_jobtext[0])
        '''
        #context = {'scoreval': scoreval}
        total_time = ' : '+str(round((time.time() - start_time1),2))+' seconds'
        context = {'total_time':total_time,'data':predictiondf_html}

    return render(request, 'predict_all.html', context)


def visualize(request):
    if request.method == "GET":
        # index of the resume for which further analysis and comparison is required
        idx = 0  # please change the index as per your requirement.
        #compare_df = input_resume_df  # X_train # we want to compare the results from test data thus using X_test, it can be changed to X_train
        compare_df = pd.read_csv('A:/UOH_Diploma_AI_ML/projects/ml_apps/resume_prediction_ml_apps/input_data/input_resume_df.csv')
        resume_id = compare_df.loc[idx, 'CandidateID']

        job_description_df, tokenized_jobtext = clean_jd()

        print(colored('Comparison Between Candidate {} and Job Description'.format(resume_id), 'green', attrs=['bold']))
        print(colored('Word Cloud For Candidate {}'.format(resume_id), 'blue', attrs=['bold']))
        matching_words,matching_word_length,matching_word_length_percentage,input_wordcloud = compare_jd_vs_resume(idx, compare_df, tokenized_jobtext)
        print(colored('Word Cloud For Job Description', 'blue', attrs=['bold']))
        jd_wordcloud = plot_wordcloud(tokenized_jobtext[0])
        messages.success(request, 'Your message has been sent!')
        scorevaldf = pd.read_csv('A:/UOH_Diploma_AI_ML/projects/ml_apps/resume_prediction_ml_apps/score/scoredf.csv')
        scoreval = scorevaldf['score'][0]
        context = {'matching_word_length_percentage': matching_word_length_percentage,
                   'matching_word_length': matching_word_length,
                   'matching_words': matching_words,
                   'input_wordcloud': input_wordcloud,
                   'jd_wordcloud': jd_wordcloud,
                   'scoreval': scoreval}

    return render(request, 'visualize.html',context)