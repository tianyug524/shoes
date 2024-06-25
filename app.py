import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import streamlit as st
import pickle
import pandas as pd

# 加载训练好的模型
model_filename = 'resNet34_6_2_2.pkl'  # 修改为你的模型路径
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)
loaded_model.eval()  # 设置为评估模式

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 读取图像并预处理
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度
    return img_tensor


# 提取特征向量
def extract_features(img_path):
    img_tensor = load_and_preprocess_image(img_path)
    with torch.no_grad():
        features = loaded_model(img_tensor.to(device))
    return features.cpu().numpy().flatten()


# 计算相似性
def compute_similarity(input_features, db_features):
    similarities = cosine_similarity([input_features], db_features)
    return similarities


# 加载数据库中的所有鞋子图像特征向量
def load_database_features(db_dir):
    db_features = []
    db_paths = []
    for category in os.listdir(db_dir):
        category_path = os.path.join(db_dir, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                features = extract_features(img_path)
                db_features.append(features)
                db_paths.append(img_path)
    return np.array(db_features), db_paths


# 推荐相似的鞋子
def recommend_similar_shoes(input_img_path, db_paths, db_features, top_n=5):
    input_features = extract_features(input_img_path)
    similarities = compute_similarity(input_features, db_features)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    recommended_paths = [db_paths[i] for i in top_indices]
    recommended_similarities = [similarities[0][i] for i in top_indices]
    return recommended_paths, recommended_similarities


# 显示推荐结果
def display_recommendations(input_img_path, recommended_paths, recommended_similarities):
    st.image(input_img_path, caption='Input Image', use_column_width=True)
    st.write("Recommended Shoes:")
    scores = []
    for i, (path, sim) in enumerate(zip(recommended_paths, recommended_similarities)):
        st.image(path, caption=f'Recommendation {i + 1} - Similarity: {sim:.4f}', use_column_width=True)
        score = st.slider(f'Rate Recommendation {i + 1}', min_value=1, max_value=5, step=1, key=f'score_{i}')
        scores.append(score)
    return scores


# 保存评分结果
def save_ratings(input_img_path, recommended_paths, scores):
    ratings_file = 'ratings.csv'
    ratings_data = []
    for path, score in zip(recommended_paths, scores):
        ratings_data.append([input_img_path, path, score])
    ratings_df = pd.DataFrame(ratings_data, columns=['input_image', 'recommended_image', 'rating'])
    if os.path.exists(ratings_file):
        ratings_df.to_csv(ratings_file, mode='a', header=False, index=False)
    else:
        ratings_df.to_csv(ratings_file, mode='w', header=True, index=False)


# Streamlit UI
st.title("Shoe Recommendation System")
st.write("Upload an image of a shoe and get recommendations for similar shoes.")

# 数据库图像路径
db_dir = './data/shoe'  # 数据库图像文件夹

# 加载数据库特征向量
db_features, db_paths = load_database_features(db_dir)

# 用户上传图片
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # 保存上传的图片
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 获取推荐的相似鞋子
    recommended_paths, recommended_similarities = recommend_similar_shoes("temp.jpg", db_paths, db_features)

    # 显示推荐结果并获取用户评分
    scores = display_recommendations("temp.jpg", recommended_paths, recommended_similarities)

    if st.button('Submit Ratings'):
        save_ratings("temp.jpg", recommended_paths, scores)
        st.write("Ratings submitted successfully!")
