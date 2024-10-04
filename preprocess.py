from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2

text = ['Будет солнечно',
        'Солнечно до конца дня. В Красносельском районе как всегда ветер',
        'Сегодня будет солнечно, ясное небо, ни облачка не появится. Ночь чистая, можно будет даже звёзды увидеть',
        'Температура опустится сразу на 6-7 градусов, так что на улице днем будет не выше плюс 13. Ощущение холода будут усиливать дожди, которые начнутся в последний день этой недели и затем будут идти несколько суток',
        'Сегодня ясно, без осадков. Атмосферное давление выше нормы.',
        'Сегодня довольно теплая для сентября погода. Небо ясное и осадков не ожидается. Так что прогноз на сегодня очень приятный.',
        'В Санкт-Петербурге сегодня ожидается +20..+18 °C, без осадков, легкий ветер.',
        'Трудно с ходу назвать фильм, который породил столько же абсурдных пародий. Данное творение несмотря на свою «трешовость», снискал огромное количество поклонников',
        'Трагикомедия, которая показывает последние дни двух молодых людей со смертельными диагнозами',
        'Фильм явно тянет на премию Оскар, проработанный сюжет, очень красивая, завораживающая картинка и я верю, что он не умер в конце фильма',
        'Еще долго после просмотра фильма я и мои друзья повторяли такие легендарные фразы главного героя, как «водка внутри — снаружи бутылка» и «ю вонна плей — лэтс плей», не забывая при этом делать губы «бантиком»']
tags = ['NOUN']
#, 'COMP',  'PRTF', 'PRTS', 'GRND', 'ADVB','VERB', 'INFN'
#, 'ADJF', 'ADJS'
for i in range(len(text)):
    text[i] = text[i].lower()
    text[i] = word_tokenize(text[i], language='russian')
    

'''filtered_text = []
for i in range(len(text)):
    filtered_text.append([])
    for j in range(len(text[i])):
        if text[i][j] not in stopwords.words("russian"):
            filtered_text[i].append(text[i][j])'''

filtered_text = text
filtered_lem = []
morph = pymorphy2.MorphAnalyzer()
for i in range(len(filtered_text)):
    filtered_lem.append([])
    for j in range(len(filtered_text[i])):
        par = morph.parse(filtered_text[i][j])[0]
        if par.tag.POS  in tags:
            filtered_lem[i].append(par.normal_form)

corpus = []
for i in range(len(filtered_lem)):
    if filtered_lem[i]:
        corpus.append([])
        corpus[-1] = filtered_lem[i][0]
        for j in range(1,len(filtered_lem[i])):
            corpus[-1] += " " + filtered_lem[i][j]

file = open("2.txt", "w")
for i in range(len(filtered_lem)-1):
    if filtered_lem[i]:
        file.write(' '.join(filtered_lem[i]) + '\n')
file.write(' '.join(filtered_lem[-1]))
file.close()