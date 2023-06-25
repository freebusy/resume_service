import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import demoji
import difflib
from razdel import sentenize
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
nltk.download('stopwords')
nltk.download('wordnet')
import matplotlib.pyplot as plt
import pickle


stop_words = set(stopwords.words('russian'))
lemmatizer = WordNetLemmatizer()


def get_unnecessary_text(row):
    """
    Берет строку датафрейма, сравнивает ДО/У/ТС/Пр 
    и возвращет неиспользуемый текст
    """
    if len(row['responsibilities']) < 1:
        return ""
    
    # пока задаем так
    unnecessary_text = row['responsibilities']

    # вытаскиваем из unnecessary_text - текст (requirements)
    if len(row['requirements']) > 8:
        diff = difflib.ndiff(unnecessary_text.split(), row['requirements'].split())
        unnecessary_text = ' '.join([word[2:] for word in diff if word.startswith('- ')])

    # вытаскиваем из unnecessary_text - текст (terms)
    if len(row['terms']) > 8:
        diff = difflib.ndiff(unnecessary_text.split(), row['terms'].split())
        unnecessary_text = ' '.join([word[2:] for word in diff if word.startswith('- ')])

    # вытаскиваем из unnecessary_text - текст (notes)
    if len(row['notes']) > 8:
        diff = difflib.ndiff(unnecessary_text.split(), row['notes'].split())
        unnecessary_text = ' '.join([word[2:] for word in diff if word.startswith('- ')])

    return unnecessary_text


def del_emoji(text):
    text = demoji.replace(str(text), "")
    text = text.replace("✔", "").replace("▉", "")
    return text


def split_sentences(text):
    return [sentence.text for sentence in sentenize(text)]


def get_df_with_labels(input_series, label):
    combined_list = sum(input_series.tolist(), [])
    res_df = pd.DataFrame(
        {'text': combined_list, 'label': label}
    )
    res_df = res_df[
        (res_df['text'] != "") &
        (res_df['text'] != " ") &
        (res_df['text'] != "-") &
        (res_df['text'] != "\\N")
    ]
    
    return res_df


def get_df_for_training(df):

    df_1 = get_df_with_labels(
        df['unnecessary_text_split'],
        label='null_class'
    )

    df_2 = get_df_with_labels(
        df['requirements_split'],
        label='requirements'
    )

    df_3 = get_df_with_labels(
        df['terms_split'],
        label='terms'
    )

    df_4 = get_df_with_labels(
        df['notes_split'],
        label='notes'
    )

    res_df = pd.concat([df_1, df_2, df_3, df_4]).reset_index(drop=True)
    return res_df


def preprocess_text(text):

    # к нижнему регистру
    text = text.lower()

    # перенос каретки
    del_n = re.compile('\n')               
    # html-теги
    del_tags = re.compile('<[^>]*>')        
    # содержимое круглых скобок
    del_brackets = re.compile('\([^)]*\)')  
    
    # чистим с помощью регулярок
    text = del_n.sub(' ', text)
    text = del_tags.sub('', text)
    text = del_brackets.sub('', text)

    # токенизация
    tokens = word_tokenize(text)

    # удаление стоп-слов
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

    # лемматизация
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # объединение токенов обратно в предложение
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def merge_responsibilities(group):
    sorted_group = group.sort_values('index_sentence')
    merged_responsibilities = ' '.join(sorted_group['responsibilities_sentence'])
    return merged_responsibilities
    

def main():
    
    # Загрузка данных из файла Excel
    df = pd.read_excel('../data/data_train.xlsx', sheet_name='МАКСУ')

    # отбор колонок
    cols = [
        # 'id',
        'responsibilities(Должностные обязанности)',
        'requirements(Требования к соискателю)',
        'terms(Условия)',
        'notes(Примечания)'
    ]
    df = df[cols].set_index(pd.Index(range(2, len(df) + 2)))

    # переименование для удоства
    df = df.rename(
        columns={
            'responsibilities(Должностные обязанности)': 'responsibilities',
            'requirements(Требования к соискателю)': 'requirements',
            'terms(Условия)': 'terms',
            'notes(Примечания)': 'notes'
        }
    )

    # заполнение NaN
    df['responsibilities'] = df['responsibilities'].fillna('')
    df['requirements'] = df['requirements'].fillna('')
    df['terms'] = df['terms'].fillna('')
    df['notes'] = df['notes'].fillna('')

    # удаление смайлов и посторонних символов - препроцессинг по всем нужным колонкам
    # df = df.applymap(preprocess_text)

    df = df.applymap(del_emoji)

    # текст, который не используется в колонках (null_class)
    df['unnecessary_text'] = df.apply(
        get_unnecessary_text,
        axis=1
    )

    # # выделение предложений
    df['unnecessary_text_split'] = df['unnecessary_text'].apply(split_sentences)
    # df['responsibilities_split'] = df['responsibilities'].apply(split_sentences)
    df['requirements_split'] = df['requirements'].apply(split_sentences)
    df['terms_split'] = df['terms'].apply(split_sentences)
    df['notes_split'] = df['notes'].apply(split_sentences)

    # делаем датасет для тренировки (классификация)
    df_for_training = get_df_for_training(
        df[[
            'unnecessary_text_split', 
            'requirements_split', 
            'terms_split', 
            'notes_split'
        ]]
    )

    # препроцессинг
    df_for_training['text'] = df_for_training['text'].apply(preprocess_text)

    # TF-IDF
    # tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer = TfidfVectorizer(
        # ngram_range=(2, 4),
        # analyzer='char_wb'
        # sublinear_tf=True,
        # use_idf=False,
        # norm='l1'
        # norm=None
    )

    # разбили данные train / test
    X_train, X_test, y_train, y_test = train_test_split(
        df_for_training['text'], 
        df_for_training['label'], 
        test_size=0.10, 
        random_state=42
    )

    # обучаем и применяем tfidf
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print('старт обучения')
    # тренируем модель 
    model = SVC(C=1, kernel='linear')
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Accuracy:", accuracy)
    print("f1_score:", f1)
    print()
    print(classification_report(y_test, y_pred))
    print()
    print('Матрица ошибок')
    print(confusion_matrix(y_test, y_pred))
    print('сохранение модели (и tfidf) в папку models')

    # Сохранение модели в файл с помощью pickle
    with open('../models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # cохранение модели TfidfVectorizer с помощью pickle
    with open('../models/vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)

    print('[DONE]')


if __name__ == '__main__':
    # запуск
    main()