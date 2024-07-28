import numpy as np
import pandas as pd
import streamlit as st
import io
import imageio
import requests
from PIL import Image
import torch 
import torch.nn as nn
import json

from models.preprocess import preprocess



# --------- double --------------

from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import resnet50
from torchvision.models import resnet18

model_1 = vgg19(weights=VGG19_Weights.DEFAULT)
model_2 = resnet18(pretrained=True)
def double_classify(img): 
    model_1.eval()
    model_2.eval()
    pred1 = model_1(img.unsqueeze(0)).softmax(dim=1)
    pred2 = model_2(img.unsqueeze(0)).softmax(dim=1)
    pred_vote = (pred1 + pred2)/2
    sorted, indices = torch.sort(pred_vote, descending=True)
    top_5 = (sorted*100).tolist()[0][:5]
    top_5_i = indices.tolist()[0][:5]
    top_5_n = list(map(decode, top_5_i))
    return top_5_n, top_5
labels = json.load(open('models/coffe_imagenet_class_index.json'))
decode = lambda x: labels[str(x)][1]



# ----------- Streamlit --------------------------


st.title('ПРОЕКТ:Введение в нейронные сети. Определение агро-культур и цвета зёрен кофе')

st.sidebar.header('Выберите страницу')
page = st.sidebar.radio("Выберите страницу", ["Общая информация", "Кофе", "Агрокультуры", "ТОП-5 предсказанных категорий"])

if page == "Общая информация":
        
        st.header('Задачи:')
        st.subheader('*Задача №1*: Классификация кофе')
        

        st.subheader('*Задача №2*: Классификация агро-культур')
        

        st.subheader('*Задача №3*: Классификация всего подряд')
        

        st.header('Команда "VGG":')
        st.subheader('Валерия')
        st.subheader('Владислав')


if page == "Кофе":

    # -------- coffee -------------


    from torchvision.models import resnet18, ResNet18_Weights
    model_coffe = resnet18(weights=ResNet18_Weights.DEFAULT)
    model_coffe.fc = nn.Linear(512,4)
    model_coffe.load_state_dict(torch.load('models/coffee_save.pt', map_location=torch.device('cpu')))
    model_coffe.eval()
    coffee_dict = {0: 'Dark', 1: 'Green', 2: 'Light', 3: 'Medium'}

    # image_url = st.text_input("Введите URL картинки кофейного зерна")
    image = None

    # if image_url:
    #             response = requests.get(image_url)
    #             image = Image.open(io.BytesIO(response.content))
    #             st.subheader('Загруженная картинка')
    #             st.image(image)

    uploaded_file = st.file_uploader("Перетащите картинку сюда или кликните для выбора файла", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.subheader('Загруженная картинка')
                st.image(image)

    
    if image is not None:
        image = preprocess(image)
        prediction = model_coffe(image.unsqueeze(0)).softmax(dim=1).argmax().item()
        st.write('Предсказанный вид кофе: ', coffee_dict[prediction])
    # chart_data = pd.DataFrame(prob_pred)

    

if page == "Агрокультуры":
    
    # --------- agriculture ---------------

    # Определение модели
    from torchvision.models import resnet18, ResNet18_Weights
    model_agri = resnet18(pretrained=True)
    model_agri.fc = nn.Linear(512,30)
    model_agri.load_state_dict(torch.load('models/agricultural_save_18.pt', map_location=torch.device('cpu')))
    model_agri.eval()
    agri_dict = {0: 'gram', 1: 'sugarcane', 2: 'Tobacco-plant', 3: 'Lemon', 4: 'rice', 5: 'Pearl_millet(bajra)', 6: 'cotton', 7: 'cucumber', 8: 'chilli', 9: 'cherry', 10: 'cardamom', 11: 'tea', 12: 'jowar', 13: 'Olive-tree', 14: 'wheat', 15: 'vigna-radiati(Mung)', 16: 'coconut', 17: 'Fox_nut(Makhana)', 18: 'almond', 19: 'clove', 20: 'coffee-plant', 21: 'mustard-oil', 22: 'jute', 23: 'banana', 24: 'soyabean', 25: 'papaya', 26: 'pineapple', 27: 'tomato', 28: 'sunflower', 29: 'maize'}

    # image_url = st.text_input("Введите URL картинки агрокультуры")
    image = None

    # if image_url:
    #             response = requests.get(image_url)
    #             image = Image.open(io.BytesIO(response.content))
    #             st.subheader('Загруженная картинка')
    #             st.image(image)
    
    uploaded_file = st.file_uploader("Перетащите картинку сюда или кликните для выбора файла", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.subheader('Загруженная картинка')
                st.image(image)
    
    if image is not None:
        image = preprocess(image)
        prediction = model_agri(image.unsqueeze(0)).softmax(dim=1).argmax().item()
        st.write('Предсказанный вид агрокультуры: ', agri_dict[prediction])
    # chart_data = pd.DataFrame(prob_pred)

if page == "ТОП-5 предсказанных категорий":
    
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.models import vgg19, VGG19_Weights
    model_1 = vgg19(weights=VGG19_Weights.DEFAULT)
    model_2 = resnet50(pretrained=True)
    def double_classify(img): 
        model_1.eval()
        model_2.eval()
        pred1 = model_1(img.unsqueeze(0)).softmax(dim=1)
        pred2 = model_2(img.unsqueeze(0)).softmax(dim=1)
        pred_vote = (pred1 + pred2)/2
        sorted, indices = torch.sort(pred_vote, descending=True)
        top_5 = (sorted*100).tolist()[0][:5]
        top_5_i = indices.tolist()[0][:5]
        top_5_n = list(map(decode, top_5_i))
        return top_5_n, top_5
    labels = json.load(open('models/coffe_imagenet_class_index.json'))
    decode = lambda x: labels[str(x)][1]
    
    image = None
    # image_url = st.text_input("Введите URL изображения чего угодно")

    # if image_url:
    #             response = requests.get(image_url)
    #             image = Image.open(io.BytesIO(response.content))
    #             st.subheader('Загруженная картинка')
    #             st.image(image)

    uploaded_file = st.file_uploader("Перетащите картинку сюда или кликните для выбора файла", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.subheader('Загруженная картинка')
                st.image(image)

    if image is not None:
            image = preprocess(image)
            classes_pred, prob_pred = double_classify(image)
            for i in range(5): 
                    st.write(f'С вероятностью {prob_pred[i]}% это {classes_pred[i]}')
    # chart_data = pd.DataFrame(prob_pred)  
    # chart_data = pd.DataFrame(prob_pred)