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


def get_df_for_training():

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
    # загрузка
    df = pd.read_excel('../data/data.xlsx')

    # переименование для удоства
    df = df.rename(
        columns={
            'responsibilities(Должностные обязанности)': 'responsibilities',
            'requirements(Требования к соискателю)': 'requirements',
            'terms(Условия)': 'terms',
            'notes(Примечания)': 'notes',
        }
    )

    # заполнение NaN
    df['responsibilities'] = df['responsibilities'].fillna('')
    df['requirements'] = df['requirements'].fillna('')
    df['terms'] = df['terms'].fillna('')
    df['notes'] = df['notes'].fillna('')

    # удаление смайлов и посторонних символов - препроцессинг по всем нужным колонкам
    df = df.applymap(del_emoji)
    # разбиение всех текстов на отдельные предложения
    df['responsibilities_split'] = df['responsibilities'].apply(split_sentences)

    # создадим датафрейм для хранения отдельных предложений
    sentences_df = pd.DataFrame(columns=['id_text', 'responsibilities_sentence', 'index_sentence'])

    # пройдем по всем строкам исходного датафрейма и вытащим отдельные предложения
    for index, row in df.iterrows():
        id_text = row['id']
        responsibilities_sentence = row['responsibilities_split']

        # проход по каждому предложению
        for index_sentence, sentence in enumerate(responsibilities_sentence):
            # добавление предложения в датафрейм
            new_row = {
                'id_text': id_text, 
                'responsibilities_sentence': sentence, 
                'index_sentence': index_sentence
            }
            sentences_df.loc[len(sentences_df)] = new_row

    # препроцессинг
    sentences_df['text'] = sentences_df['responsibilities_sentence'].apply(preprocess_text)

    # Загрузка модели из файла
    with open('../models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # загрузка модели TfidfVectorizer
    with open('../models/vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    # применение модели к датафрейму
    X_tfidf = tfidf_vectorizer.transform(sentences_df['text'])
    y_predict = model.predict(X_tfidf)

    sentences_df['labels'] = y_predict    
    
    # группируем по индексу текста и по лейблу(колонке в дальнейшем)
    # и сортируем по порядку предложений
    df_by_group = sentences_df.groupby(['id_text', 'labels']).apply(
        merge_responsibilities
    ).reset_index(name='text')

    # делаем нужную форму датафрейма
    df_encoded = pd.get_dummies(df_by_group['labels'])
    merged_df = pd.concat([df_by_group[['id_text', 'text']], df_encoded], axis=1)
    # заполняем будующие колонки текстом
    merged_df['notes'] = merged_df.apply(lambda row: row['text'] if row['notes'] == 1 else '', axis=1)
    merged_df['null_class'] = merged_df.apply(lambda row: row['text'] if row['null_class'] == 1 else '', axis=1)
    merged_df['requirements'] = merged_df.apply(lambda row: row['text'] if row['requirements'] == 1 else '', axis=1)
    merged_df['terms'] = merged_df.apply(lambda row: row['text'] if row['terms'] == 1 else '', axis=1)

    # объединяем в "тексты" несколько предложений
    grouped_df = merged_df.groupby('id_text').sum().reset_index()
    # возьмем нужные колонки в нужном порядке
    grouped_df = grouped_df[['id_text', 'requirements', 'terms', 'notes','null_class']]
    # переименуем для мержа
    grouped_df.rename(columns={
        'requirements': 'requirements(Требования к соискателю)',
        'terms': 'terms(Условия)',
        'notes': 'notes(Примечания)', 
        'null_class': 'null_class(остатки от Долж обяз)',
        'id_text': 'id'
    }, inplace=True)
    # приведение к типу int
    grouped_df['id'] = grouped_df['id'].astype(int)

    # загрузка данных для дальнейшего мержа
    df = pd.read_excel('../data/data.xlsx')

    # удалим колонки, чтобы не дублировались
    df.drop(
        [
            'notes(Примечания)', 
            'requirements(Требования к соискателю)', 
            'terms(Условия)'
        ], 
        axis=1, 
        inplace=True
    )
    
    # соединяем по id
    full_df = df.merge(grouped_df, on='id', how='left')
    # сохраняем в /data/data_results.xlsx
    full_df.to_excel('../data/data_results.xlsx')


if __name__ == '__main__':
    # запуск
    main()