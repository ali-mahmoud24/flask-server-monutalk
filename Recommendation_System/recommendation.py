
import pandas as pd
import numpy as np


data = pd.read_csv('./Recommendation_System/MY_last_data.csv',encoding='ISO-8859-1')
data

data['Tags'] = data['category'] + ','+ data['Keywords'] + ','+ data['Description'] + ','+data['State_Name']

data['Tags'][0]

data['Tags'] = data['Tags'].apply(lambda x:x.lower() )


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000,stop_words='english')

cv.fit_transform(data['Tags']).toarray()

vectors = cv.fit_transform(data['Tags']).toarray()


import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)

data['Tags']=data['Tags'].apply(stem)

vectors = cv.fit_transform(data['Tags']).toarray()


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

import os
import pickle

if isinstance(vectors, np.ndarray):
    vectors = pd.DataFrame(vectors)

# Create a directory to store the similarity\
os.makedirs('./Recommendation_System/similarity_matrices', exist_ok=True)

# Loop all states
for category in data['category'].unique():
    category_filtered_museums = data[data['category'] == category]
    category_vectors = vectors.loc[category_filtered_museums.index]
    category_similarity = cosine_similarity(category_vectors)

    # print(f"Category: {category}")
    # print(f"Category_vectors shape: {category_vectors.shape}")
    # print(f"Category_similarity shape: {category_similarity.shape}")

    filename = os.path.join('./Recommendation_System/similarity_matrices', f'similarity_{category.replace(" ", "_")}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(category_similarity, f)

category_name = data[data['name']=='National Museum of Egyptian Civilization NEMC'].iloc[0].category

museum_index = list(data[(data['category'] =='Speciality Museums')].index)

category_name = 'Speciality Museums'

filename = f'./Recommendation_System/similarity_matrices/similarity_{category_name.replace(" ", "_")}.pkl'
with open(filename, 'rb') as f:
    category_similarity = pickle.load(f)

category_name = data[data['name']=='National Museum of Egyptian Civilization NEMC'].iloc[0].category
filename = f'./Recommendation_System/similarity_matrices/similarity_{category_name.replace(" ", "_")}.pkl'
with open(filename, 'rb') as f:
    category_similarity = pickle.load(f)


museum_index = data[(data['category'] == category_name) &
                           (data['name'] == 'National Museum of Egyptian Civilization NEMC')].index[0]

local_index = list(data[data['category'] == category_name].index).index(museum_index)
distances = category_similarity[local_index]

# The number of museums that are most similar should not exceed the size of the matrix
num_of_museums = category_similarity.shape[0]
num_of_recommendations = min(num_of_museums, 5)

def recommend(museum_name):
    if museum_name not in data['name'].values:
          print(f"Museum named '{museum_name}' not found.")
          return []
    category_name = data[data['name']==museum_name].iloc[0].category
    filename = f'./Recommendation_System/similarity_matrices/similarity_{category_name.replace(" ", "_")}.pkl'
    with open(filename, 'rb') as f:
        category_similarity = pickle.load(f)


    museum_index = data[(data['category'] == category_name) &
                           (data['name'] == museum_name)].index[0]

    local_index = list(data[data['category'] == category_name].index).index(museum_index)
    distances = category_similarity[local_index]

    # The number of museums that are most similar should not exceed the size of the matrix
    num_of_museums = category_similarity.shape[0]
    num_of_recommendations = min(num_of_museums, 5)

    # Get an index of the most similar museums (make sure not to go out of range)
    similar_indices = sorted(range(num_of_museums), key=lambda i: distances[i], reverse=True)[1:num_of_recommendations+1]

    category_filtered_museums = data[data['category'] == category_name]
    recommended_museum_names = category_filtered_museums.iloc[similar_indices]['name'].tolist()
    return recommended_museum_names


def get_museums():
  return data['name'].values.tolist()

# Function to get the index of a given name
def get_index(name):
    result = np.where(data == name)
    return result[0][0] if result[0].size > 0 else None

# Example usage
name_value = "The Museum of Egyptian Antiquities"  # Replace with the actual name you want to find
index = get_index(name_value)
# print(f"The index for '{name_value}' is {index}.")

def get_recommendations(museum_name):
  museums = recommend(museum_name)

  recommended_museums = []
  index = get_index(museum_name)

  museum_name = data['name'][index]
  museum_category = data['category'][index]
  museum_description = data['Description'][index]
  museum_location = data['State_Name'][index]

  search_result = { 'name' : museum_name , 'category' : museum_category , 'description' : museum_description , 'location' : museum_location}


  for museum_name in museums:
    index = get_index(museum_name)

    museum_name = data['name'][index]
    museum_category = data['category'][index]
    museum_description = data['Description'][index]
    museum_location = data['State_Name'][index]

    museum = { 'name' : museum_name , 'category' : museum_category , 'description' : museum_description , 'location' : museum_location}


    recommended_museums.append(museum)

  return recommended_museums, search_result

import pickle
#Converting DF into dictionary for dumping the data

pickle.dump(data,open('./Recommendation_System/museums.pkl','wb'))

pickle.dump(data.to_dict(),open('./Recommendation_System/museums_dict.pkl','wb'))

pickle.dump(similarity,open('./Recommendation_System/similarity.pkl','wb'))

# Adding State_Name filter to the app

unique_states_df = pd.DataFrame(data['category'].unique(), columns=['category'])

unique_states_df.to_pickle('./Recommendation_System/category.pkl')