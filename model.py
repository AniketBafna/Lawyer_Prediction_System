#from distutils.command.clean import clean
import pickle
import re
import nltk
import streamlit as st
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

# Loading data
df = pd.read_csv("new_final.csv")
df.drop(columns=['Unnamed: 0','case_text'], inplace=True)

# loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def textcleaning(text):
    import string
    from textblob import TextBlob
    exclude = string.punctuation
    
    cleantext = text.lower()     # lower the string
    cleantext = re.sub("<.*?>", " ", cleantext)   # Removing the HTML tags
    cleantext = re.sub("https?://\S+|www\.\S+"," ",cleantext)   # Removing the URLS's
    cleantext = cleantext.translate(str.maketrans('','',exclude)) # Removing the Punctuations
    cleantext = re.sub("\s+"," ",cleantext) # Removing the extra characters (\r\n)
    #cleantext = TextBlob(cleantext)
    #cleantext = cleantext.correct().string  # Removing the Spelling mistake
    return cleantext


# web app
def model():
    st.title("Lawyer Recommendation System")
    upload_file = st.file_uploader("Upload Resume",type=['txt','pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')


        clean_resume = textcleaning(resume_text)
        clean_resume = tfidf.transform([clean_resume])
        prediction_id = clf.predict(clean_resume)[0]

        category_mapping = {49: 'Pooja Singh',81: 'Suman Mishra',7: 'Arjun Malhotra',47: 'Pooja Reddy',82: 'Sunita Chopra',22: 'Karan Verma',87: 'Vijay Bose',95: 'Vivek Nair',8: 'Arjun Sharma',69: 'Ramesh Patel',91: 'Vivek Bose',44: 'Pooja Mehta',76: 'Sita Gandhi',39: 'Neeraj Malhotra',96: 'Vivek Reddy',15: 'Karan Gandhi',19: 'Karan Pandey',5: 'Arjun Agarwal',12: 'Asha Pandey',50: 'Priya Chopra',86: 'Sushma Sharma',26: 'Manoj Joshi',0: 'Anil Khan',20: 'Karan Rao',35: 'Meena Sharma',97: 'Vivek Saxena',30: 'Manoj Singh',38: 'Neeraj Joshi',24: 'Manoj Gandhi',78: 'Sita Pandey',63: 'Raj Mehta',98: 'Vivek Sharma',3: 'Anil Singh',32: 'Meena Gandhi',71: 'Ravi Khan',93: 'Vivek Gandhi',54: 'Priya Mishra',27: 'Manoj Nair',85: 'Sushma Rao',2: 'Anil Mehta',11: 'Asha Joshi',75: 'Sita Bose',28: 'Manoj Pandey',88: 'Vijay Khan',17: 'Karan Malhotra',45: 'Pooja Pandey',46: 'Pooja Patel',74: 'Rekha Patel',13: 'Asha Sharma',66: 'Raj Singh',9: 'Asha Bose',29: 'Manoj Patel',33: 'Meena Joshi',80: 'Suman Malhotra',10: 'Asha Gandhi',94: 'Vivek Mishra',16: 'Karan Khan',58: 'Rahul Mishra',99: 'Vivek Singh',90: 'Vijay Saxena',23: 'Manoj Chopra',72: 'Ravi Singh',48: 'Pooja Sharma',55: 'Priya Saxena',31: 'Meena Agarwal',1: 'Anil Malhotra',14: 'Karan Bose',59: 'Rahul Pandey',79: 'Suman Gandhi',21: 'Karan Reddy',60: 'Raj Chopra',37: 'Neeraj Gandhi',25: 'Manoj Iyer',67: 'Ramesh Chopra',53: 'Priya Malhotra',92: 'Vivek Chopra',6: 'Arjun Das',83: 'Sunita Mehta',61: 'Raj Iyer',18: 'Karan Mishra',62: 'Raj Khan',41: 'Pooja Chopra',4: 'Anil Verma',73: 'Rekha Agarwal',56: 'Rahul Gandhi',40: 'Neeraj Reddy',36: 'Meena Verma',42: 'Pooja Iyer',68: 'Ramesh Pandey',70: 'Ravi Iyer',52: 'Priya Khan',65: 'Raj Rao',43: 'Pooja Khan',84: 'Sushma Chopra',51: 'Priya Gandhi',77: 'Sita Malhotra',34: 'Meena Nair',89: 'Vijay Pandey',64: 'Raj Patel',57: 'Rahul Malhotra'}
        category_name = category_mapping.get(prediction_id,"Unknown")

        #st.write("Lawyer Suggested: ", category_name)
        index =  df[df['Lawyer_name'] == prediction_id].index[0:6]

        
        
    if st.button("Show Recommendation"):
        #st.write(index)
        #recommended_lawyer_names, recommended_laywer_yoe, recommended_lawyer_total_cases, recommended_lawyer_success_cases, recommended_lawyer_fees = recommend(index)
        col1, col2, col3, col4, col5 = st.columns(5)
        num = df[df['Lawyer_name'] == prediction_id][0:6]
        def recommend(df):
            recommended_lawyer_names = []
            recommended_laywer_yoe = []
            recommended_lawyer_total_cases = []
            recommended_lawyer_success_cases = []
            recommended_lawyer_fees = []

            for index, col in num.iterrows():
                recommended_lawyer_names.append(category_name)
                recommended_laywer_yoe.append(col['years_of_experience'])
                recommended_lawyer_total_cases.append(col['no_of_cases'])
                recommended_lawyer_success_cases.append(col['no_of_success_cases'])
                recommended_lawyer_fees.append(col['fees_per_hearing'])

                #col['Lawyer_name'], col['years_of_experience'], col['no_of_cases'], col['no_of_success_cases'], col['fees_per_hearing']

            return recommended_lawyer_names, recommended_laywer_yoe, recommended_lawyer_total_cases, recommended_lawyer_success_cases, recommended_lawyer_fees

        #st.write(recommended_lawyer_names)
        #st.write(recommended_laywer_yoe)
        recommended_lawyer_names, recommended_laywer_yoe, recommended_lawyer_total_cases, recommended_lawyer_success_cases, recommended_lawyer_fees = recommend(df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("download.png")
            st.subheader(recommended_lawyer_names[0])
            st.write("Year of Experience:",recommended_laywer_yoe[0])
            st.write("Total Cases : ", recommended_lawyer_total_cases[0])
            st.write("Cases Won : ", recommended_lawyer_success_cases[0])
            st.write("Fees Per Hearing : ", recommended_lawyer_fees[0])
        with col2:
            st.image("download.png")
            st.subheader(category_mapping.get(df['Lawyer_name'].sample(n=1).values[0],"Unknown"))
            st.write("Year of Experience:",recommended_laywer_yoe[1])
            st.write("Total Cases : ", recommended_lawyer_total_cases[1])
            st.write("Cases Won : ", recommended_lawyer_success_cases[1])
            st.write("Fees Per Hearing : ", recommended_lawyer_fees[1])
        with col3:
            st.image("download.png")
            st.subheader(category_mapping.get(df['Lawyer_name'].sample(n=1).values[0],"Unknown"))
            st.write("Year of Experience:",recommended_laywer_yoe[2])
            st.write("Total Cases : ", recommended_lawyer_total_cases[2])
            st.write("Cases Won : ", recommended_lawyer_success_cases[2])
            st.write("Fees Per Hearing : ", recommended_lawyer_fees[2])
        with col1:
            st.image("download.png")
            st.subheader(category_mapping.get(df['Lawyer_name'].sample(n=1).values[0],"Unknown"))
            st.write("Year of Experience:",recommended_laywer_yoe[3])
            st.write("Total Cases : ", recommended_lawyer_total_cases[3])
            st.write("Cases Won : ", recommended_lawyer_success_cases[3])
            st.write("Fees Per Hearing : ", recommended_lawyer_fees[3])
        with col2:
            st.image("download.png")
            st.subheader(category_mapping.get(df['Lawyer_name'].sample(n=1).values[0],"Unknown"))
            st.write("Year of Experience:",recommended_laywer_yoe[4])
            st.write("Total Cases : ", recommended_lawyer_total_cases[4])
            st.write("Cases Won : ", recommended_lawyer_success_cases[4])
            st.write("Fees Per Hearing : ", recommended_lawyer_fees[4])

# Python main
if __name__ == "__main__":
    model()
