import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import calendar
import requests
import json
import librosa

from PIL import Image
from geopy.geocoders import Nominatim
from streamlit_js_eval import get_geolocation
from audio_recorder_streamlit import audio_recorder
from datetime import datetime



#loading data
desc_path = "interface/data/36ebirds_description_images.csv"
data = pd.read_csv(desc_path)
df = pd.DataFrame(data)

cord_path = "interface/data/36_species_coordinates.csv"
bird_coord = pd.read_csv(cord_path)
coord = pd.DataFrame(bird_coord)

#date retrieval

currentMonth = datetime.now().month
currentDay = datetime.now().day
currentYear = datetime.now().year
currentMonthName = calendar.month_name[currentMonth]


logo_col, title_col = st.columns([1,4])
with logo_col:
    logo = Image.open('interface/data/logo.png')
    st.image(logo)
with title_col:
    st.title("ChirpID")

#chirpID description
# st.write("ChirpID is a new tool to help nature lovers, avid birders, field researchers, and all curious souls to discover the birds around them. The model uses a functional model combining dense and convolutional neutral networks to classify bird species through preprocessed audio input. Simply record or upload an audio clip, and ChirpID will tell you about the bird species you are hearing!")

st.write("Discover the birds you are hearing with ChirpID! Simply: ")

st.markdown('''
            - Upload or record bird sound

            - Tell us the location of recording (or use current location)

            - Click "ID my Bird"

            Enjoy learning about the avian friends around you!

            ''')

with st.sidebar:
    audio_bytes = audio_recorder(
        text="Record a bird",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        #icon_name="fa-solid fa-dove",
        icon_size="3x",
        sample_rate=22050
    )

    upload_audio = st.file_uploader('Or: Upload sound for classification!', type=['wav','mp3'])

#chirpID location

    user_loc = st.sidebar.text_input('Enter your location below')
    if user_loc:
        try:
            #calling the Nominatim tool
            loc = Nominatim(user_agent="GetLoc")
            getLoc = loc.geocode(user_loc)
            # printing address
            user_lat = getLoc.latitude
            user_long = getLoc.longitude
            st.sidebar.write('Your Location is: ', getLoc.address)
            st.sidebar.write(f"({user_lat}, {user_long})")

        except:
            st.write("chirp error")
    if st.checkbox("Use my current location"):
        loc = get_geolocation()
        try:
            user_lat = loc.get('coords').get('latitude')
            user_long = loc.get('coords').get('longitude')
            st.sidebar.write(f"Your current location is:")
            st.sidebar.write(f"({user_lat}, {user_long})")
        except:
            st.success('Loading coordinates, please wait...')

    # if audio_bytes:
    #     st.write("WE GOT AUDIO")

def preprocess_picture(file):

    sampling_rate=22050
    Signal , sr = librosa.load(file,sr=sampling_rate)
    if Signal.shape[0] <= sr*10:
        num_zeros = int((sr*10)-Signal.shape[0])
        padding = np.zeros(num_zeros)
        Signal = np.concatenate((Signal, padding), axis=0)
    Signal = Signal[0:sr*6]


    n_fft = 2058 # the window, or kernel as I understand it
    hop_length = 128 # the amount of shifting the window to the right
    stft = librosa.core.stft(Signal , hop_length = hop_length , n_fft = n_fft)
    spectogram = np.abs(stft)
    picture = librosa.amplitude_to_db(spectogram)
    return picture

def get_image_url(predicted_species):
    return df[df['en']==predicted_species]['image_links'].squeeze()

#Define a function that retrieves the description of the brid
def get_description(predicted_species):
    return df[df['en']==predicted_species]['descriptions'].squeeze()

def get_map_data(predicted_species):
    df= coord[coord['en']==predicted_species][['lat','lng']].dropna().rename(columns={'lng':'lon'})
    pd.to_numeric(df['lat'])
    pd.to_numeric(df['lon'])
    return df


if upload_audio:
    with open("audio.wav", mode="bw") as f:
        f.write(upload_audio.read())

    log_spectogram = preprocess_picture("audio.wav")
    st.audio("audio.wav", format="audio/wav")
    fig, ax = plt.subplots(figsize=(15,3))
    librosa.display.specshow(log_spectogram , sr = 22050 , hop_length = 128, htk=True,y_axis="hz" ,x_axis="s", cmap=None)

    plt.xlabel('Time (s)', fontdict={'size':15})
    plt.ylabel('Frequency (Hz)', fontdict={'size':15})
    plt.colorbar()
    fig.savefig('spectrogram.png', bbox_inches='tight', transparent=True)

    img = Image.open('spectrogram.png')
    # st.write(log_spectogram.shape)
    st.image(img)

if audio_bytes:
    abs_fp = "audio.wav"
    with open(abs_fp, mode="bw") as f:
        f.write(audio_bytes)

    log_spectogram = preprocess_picture("audio.wav")
    st.audio("audio.wav", format="audio/wav")
    fig, ax = plt.subplots(figsize=(15,3))
    librosa.display.specshow(log_spectogram , sr = 22050 , hop_length = 128, htk=True,y_axis="hz" ,x_axis="s", cmap=None)

    plt.xlabel('Time (s)', fontdict={'size':15})
    plt.ylabel('Frequency (Hz)', fontdict={'size':15})
    plt.colorbar()
    fig.savefig('spectrogram.png', bbox_inches='tight', transparent=True)

    img = Image.open('spectrogram.png')
    # st.write(log_spectogram.shape)
    st.image(img)


elif audio_bytes:
    log_spectogram = preprocess_picture("audio.wav")
    st.audio(audio_bytes, format="audio/wav")
    fig, ax = plt.subplots(figsize=(15,3))
    librosa.display.specshow(log_spectogram , sr = 22050 , hop_length = 128, htk=True,y_axis="hz" ,x_axis="s", cmap=None)

    plt.xlabel('Time (s)', fontdict={'size':15})
    plt.ylabel('Frequency (Hz)', fontdict={'size':15})
    plt.colorbar()
    fig.savefig('spectrogram.png', bbox_inches='tight', transparent=True)

    img = Image.open('spectrogram.png')
    # st.write(log_spectogram.shape)
    st.image(img)


if st.sidebar.button('ID My Bird'):
    # try:# upload_audio or audio_bytes:

    lng = user_long
    lat = user_lat
    month = currentMonth

    url = 'https://chirpapi-niuue56uea-an.a.run.app/predict_five?lng={}&lat={}&month={}'.format(lng, lat, month)

    headers = {
        'accept': 'application/json',
    }

    fp = "audio.wav"
    files = {
        'file': (fp, open(fp, 'rb')),
    }

    response = requests.post(url, headers=headers, files=files)
    prediction = json.loads(response.content)

    preds = []
    names= ["Alpine Leaf Warbler",]
    prediction["prediction"].split(":")
    for i in prediction["prediction"].split(":")[2:-1]:
        name = i.split(",")[1].replace("\\", "")
        pred = float(i.split(",")[0])
        names.append(name)
        preds.append(pred)

    df_2 = pd.concat([pd.DataFrame(names)[0], pd.DataFrame(preds)[0]], axis=1)
    df_2.columns = ["names", "preds"]
    df_2["preds"] = df_2["preds"].replace(np.nan, 0)

    # print(df_2.iloc[35,1])
    # print(type(df_2.iloc[35,1]))
    df_preds = df_2.sort_values(by="preds", ascending=False).reset_index()
    # print(df_preds)
    # print(df_preds.loc[0,"names"])
    # print(df_preds.loc[0,"preds"])

    # print(type(df_preds.loc[0,"names"]))
    # print(type(df_preds.loc[0,"preds"]))

    predicted_species = str(df_preds.loc[0,"names"])
    predicted_species = predicted_species.strip('"')
    confidence = df_preds.loc[0,"preds"]
    # print("-----------------")



            # st.write(f"Our model predicts it's a {predicted_species}")

    st.header(predicted_species)
    st.write(f"Confidence: {confidence}%")
    left_column, right_column = st.columns(2)

    with left_column:
        # Get the URL of the selected image
        image_url = get_image_url(predicted_species)
        print("Daurian Redstart", str(predicted_species))
        print(image_url)
        print("raeasd", df.loc[25,"en"])
        print("raeasd", type(df.loc[25,"en"]))
        print(predicted_species == "Daurian Redstart")
        print(df[df['en']==predicted_species]['image_links'])
        st.image(image_url, caption=predicted_species)

    with right_column:
        st.write(get_description(predicted_species))

    df = get_map_data(predicted_species)
    st.map(df)
    fp_1 =f"/home/ginger/code/gderiddershanghai/chirp_five/interface/data/five/{predicted_species}.mp3"
    st.audio(fp_1, format="audio/wav")
    st.markdown("""
    Made possible by
    [Joyce Siu] - (https://github.com/jss-520)
    [Helena Zhu] - (https://github.com/helchu)
    [Ginger de Ridder] - (https://github.com/gderiddershanghai/)
    """)


    if float(confidence) < 50.0:
        if st.sidebar.button(f'{predicted_species} Confidence: {confidence}'):
            predicted_species = str(df_preds.loc[1,"names"])
            predicted_species = predicted_species.strip('"')
            confidence = df_preds.loc[1,"preds"]

            url = 'https://chirpapi-niuue56uea-an.a.run.app/predict_five?lng={}&lat={}&month={}'.format(lng, lat, month)

            headers = {
                'accept': 'application/json',
            }

            fp = "audio.wav"
            files = {
                'file': (fp, open(fp, 'rb')),
            }

            response = requests.post(url, headers=headers, files=files)
            prediction = json.loads(response.content)

            preds = []
            names= ["Alpine Leaf Warbler",]
            prediction["prediction"].split(":")
            for i in prediction["prediction"].split(":")[2:-1]:
                name = i.split(",")[1].replace("\\", "")
                pred = float(i.split(",")[0])
                names.append(name)
                preds.append(pred)

            df_2 = pd.concat([pd.DataFrame(names)[0], pd.DataFrame(preds)[0]], axis=1)
            df_2.columns = ["names", "preds"]
            df_2["preds"] = df_2["preds"].replace(np.nan, 0)

            # print(df_2.iloc[35,1])
            # print(type(df_2.iloc[35,1]))
            df_preds = df_2.sort_values(by="preds", ascending=False).reset_index()
            # print(df_preds)
            # print(df_preds.loc[0,"names"])
            # print(df_preds.loc[0,"preds"])

            # print(type(df_preds.loc[0,"names"]))
            # print(type(df_preds.loc[0,"preds"]))

            predicted_species = str(df_preds.loc[0,"names"])
            predicted_species = predicted_species.strip('"')
            confidence = df_preds.loc[0,"preds"]
            # print("-----------------")



                    # st.write(f"Our model predicts it's a {predicted_species}")

            st.header(predicted_species)
            st.write(f"Confidence: {confidence}%")
            left_column, right_column = st.columns(2)

            with left_column:
                # Get the URL of the selected image
                image_url = get_image_url(predicted_species)
                print("Daurian Redstart", str(predicted_species))
                print(image_url)
                print("raeasd", df.loc[25,"en"])
                print("raeasd", type(df.loc[25,"en"]))
                print(predicted_species == "Daurian Redstart")
                print(df[df['en']==predicted_species]['image_links'])
                st.image(image_url, caption=predicted_species)

            with right_column:
                st.write(get_description(predicted_species))

            df = get_map_data(predicted_species)
            st.map(df)
            fp_1 =f"interface/data/five/{predicted_species}.mp3"
            st.audio(fp_1, format="audio/wav")
