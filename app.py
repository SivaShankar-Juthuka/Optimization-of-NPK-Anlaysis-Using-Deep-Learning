import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

rf_model_path = "random_forest_model.pkl"
rnn_model_path = "simple_rnn_model.h5"

res = {}
header_container = st.container()
header_left, header_center, header_right = st.columns([30, 3, 1])

# Add content to the header
with header_right:
    st.image("static/Picture3.png", width = 180) # Adjust width as needed

with header_left:
    st.title("Optimization of NPK Analysis using Deep Learning")

dic = {
       'rice' : [79.89, 47.58, 39.87, 23.68, 82.27, 6.42], 
       'maize' : [77.76, 48.44, 19.79, 22.38920391, 65.09224945, 6.245189722], 
       'chickpea' : [40.09, 67.79, 79.92, 18.87284675, 16.86043942, 7.336956624],
       'kidneybeans' : [20.75, 67.54, 20.05, 20.11508469, 21.60535673, 5.749410586],
       'pigeonpeas' : [20.73, 67.73, 20.29,	27.74176223, 48.06163308, 5.79417488],
       'mothbeans' : [21.44, 48.01, 20.23, 28.19492048, 53.16041803, 6.831174083],
       'mungbean' : [20.99, 47.28, 19.87, 28.52577474, 85.49997454, 6.72395694],
       'blackgram' : [40.02, 67.47, 19.24, 29.97333968, 65.11842559, 7.133951629],
       'lentil' : [18.77, 68.36, 19.41, 24.5090524, 64.80478468, 6.927931572],
       'pomegranate' : [18.87, 18.75, 40.21, 21.83784172, 90.12550379, 6.429171841],
       'banana' : [100.23, 82.01, 50.05, 27.37679831, 80.35812258, 5.98389318],
       'mongo' : [20.07, 27.18, 29.92, 31.20877015, 50.1565727, 5.7663728],
       'grapes' : [23.18, 132.53, 200.11, 23.84957512, 81.87522752, 6.025936681], 
       'watermelon' : [99.42, 17, 50.22, 25.59176724, 85.16037529, 6.495778302],
       'muskmelon' : [100.32, 17.72, 50.08, 28.66306576, 92.34280196, 6.358805452],
       'apple' : [20.8, 134.22, 199.89, 22.63094241, 92.33338288, 5.929662932],
       'orange' : [19.58, 16.55, 10.01, 22.7657255, 92.17020876, 7.016957453],
       'papaya' : [49.88, 59.05, 50.04, 33.72385874, 92.40338768, 6.741442373],
       'coconut' : [21.98, 16.93, 30.59, 27.40989217, 94.84427181, 5.976562126],
       'cotton' : [117.77, 46.24, 19.56, 23.9889579, 79.84347425, 6.912675496],
       'jute' : [78.4, 46.86, 39.99, 24.95837583, 79.63986421, 6.732777568],
       'coffee' : [101.2, 28.74, 29.94, 25.54047682, 58.8698463, 6.790308275]
    }

ploting_data = {}

def display_Home():
    # Define the content for each card
    st.markdown('<div class="home-page">', unsafe_allow_html=True)
    cards = [
        {
         "title": "Nitrogen (N)", "content" :
          """ Nitrogen is a vital nutrient for plants and soil fertility. It helps plants grow and develop by:\n 
            1. Forming proteins, enzymes, and chlorophyll
            2. Supporting the development of roots 
            3. Helping plants absorb other nutrients
            4. Contributing to the accumulation of organic matter in the soil
            5. Being a key factor in determining grain yield
            6. Supporting photosynthesis """
        },
        {"title": "Potassium (K)", "content": 
         """ Potassium is a vital nutrient for plant growth and development. It is involved in many processes, including:\n
            1. Water movement: Potassium is associated with the movement of water, nutrients, and carbohydrates in plant tissue.
            2. Enzyme activation: Potassium is involved with enzyme activation within the plant, which affects protein, starch, and adenosine triphosphate (ATP) production.
            3. Cell wall construction: Potassium nitrate helps to construct thicker cell walls.
            4. Electrolyte levels: Potassium nitrate increases the level of electrolytes in the cells. """
        },
        {"title": "Phosphorus (P)", "content": 
          """ Phosphorus is a macronutrient that plays a key role in plant growth and development. It's a vital component of DNA and RNA, 
              and is especially important for capturing and converting the sun's energy into useful plant compounds. Phosphorus is involved in many plant processes, including:\n
            1. Energy transfer reactions.
            2. Development of reproductive structures.
            3. Crop maturity.
            4. Root growth.
            5. Protein synthesis. """
        },
        {"title": "Temperature", "content":
         """ Each plant species has a suitable temperature range. Within this range, higher temperatures generally promote shoot growth, including leaf expansion and stem elongation and thickening
            High temperature, even for short period, affects crop growth especially in temperate crops like wheat. High air temperature reduces the growth of shoots and in turn reduces root growth. 
            High soil temperature is more crucial as damage to the roots is severe resulting in substantial reduction in shoot growth."""
        }, 
        {"title": "Humidity", "content":
         """ Humidity can affect crops in many ways, including:\n
            1. Transpiration: Humidity levels that are too high or too low can slow down transpiration, which can inhibit plant growth, health, and development.
            2. Plant water relations: Relative humidity (RH) directly affects plant water relations.
            3. Pests: High humidity conditions can lead to more insect pests and diseases. For example, fungus gnats thrive in moist soil and feed on plant roots.
            4. Foliar and root diseases: Humid air can contribute to problems such as foliar and root diseases.
            5. Slow drying: Humid air can cause the growing medium to dry slowly.
            6. Plant stress: Humid air can cause plant stress.
            7. Loss of quality: Humid air can cause loss of quality and yield.
            8. Grain yield: Very high or very low RH is not conducive for high grain yield."""
        },
        {"title": "PH value", "content":
        """Each plant has its own recommended pH value range. The reason for this is that pH affects the availability of nutrients within the soil, and plants have different nutrient needs. 
        For example: The nutrient nitrogen, a very important plant nutrient, is readily available in soil when the pH value is above 5.5.\n
        A soil's pH (pH scale) measures its acidity or alkalinity. A pH of 7 is neutral, while a pH below 7 is acidic and a pH above 7 is alkaline. 
        Soil pH is an important factor in crop production. It affects many chemical and biochemical reactions in soil, and it also affects the availability of nutrients for plant growth.
        For most crops, a pH of 6 to 7.5 is optimal. A pH range of 5.5 to 6.5 is optimal for plant growth because it provides optimal nutrient availability."""
        },
    ]

    container = st.container()
    with container:
        for card in cards:
            st.markdown(
                f"""
                <div class="card" >
                    <div class="card-content">
                        <div class="card-text">
                            <h3>{card['title']}</h3>
                            <p>{card['content']}</p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Apply custom CSS for card styling and hover effect
    st.markdown(
        """
        <style>
        .card {
            background-size: cover;
            padding: 30px;
            border: 1px solid #ddd;
            border-radius: 25px;
            transition: transform 0.3s ease;
            margin-bottom: 20px;
        }
        .card:hover {
            transform: scale(1.10);
        }
        .card-content {
            display: flex;
            align-items: center;
        }
        .card-text {
            flex: 1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def display_Analysis():
    st.header('NPK Analysis')
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, step=0.1, format="%.2f")
    phosphorus = st.number_input("Phosphorus (P)", min_value=5.0, max_value=145.0, step=0.1, format="%.2f")
    potassium = st.number_input("Potassium (K)", min_value=5.0, max_value=205.0, step=0.1, format="%.2f")
    Temperature = st.number_input("Temperature", min_value=8.0, max_value=45.0, step=0.1, format="%.2f")
    Humidity = st.number_input("Humidity", min_value=14.0, max_value=100.0, step=0.1, format="%.2f")
    PH_Value = st.number_input("PH_Value", min_value=3.0, max_value=10.0, step=0.1, format="%.2f")
    
    if st.button("Submit"):
        tabs = st.tabs(['Result', 'Visual Representation'])
        with tabs[0]:
            process_analysis(nitrogen, phosphorus, potassium, Temperature, Humidity, PH_Value)
        with tabs[1]:
            st.subheader('Visual Representation')
            if tabs[1].is_active:
                st.empty()  # Clear the content of the visual representation tab
                display_visual()


def display_visual():
    st.write("This is the Visual Representation of NPK analysis.")
    df = pd.DataFrame(dic)
    df = df.transpose()
    legend_df = pd.DataFrame({
        "Inputs": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "PH"]
    })
    st.bar_chart((df), use_container_width=False)
    st.dataframe(legend_df, width=None)



def process_analysis(n, p, k, T, H, aph):
    """
    This Function will take inputs from user and calculate NPK analysis report.
    Parameters:
    n (float): Nitrogen in percentage
    p (float): Phosphorous in percentage
    k (float): Potassium in percentage
    T (float): Temperature in degree Celsius
    H (float): Relative humidity in percentage
    Aph (float): Acid-base indicator value
    Returns:
        None
    """
    l = [n, p, k, T, H, aph] 
    crop = solve(l)
    crop = ''.join(crop)
    display_result(crop, l)

def display_result(crop, l):
    res_container = st.container()
    st.subheader(f"For provided data {crop} is the suitable crop.")
    a = f"static/{crop}_0.jpg"
    b = f"static/{crop}_1.jpg" 
    crop_images = [a, b]
    if crop_images:
        image_col1, image_col2 = st.columns(2)
        for i, image_path in enumerate(crop_images):
            if i == 0:
                with image_col1:
                    st.image(image_path, width=1000, use_column_width='always', output_format='auto')
            else:
                with image_col2:
                    st.image(image_path, width=1000, use_column_width='always', output_format='auto')

    st.write("Recommended Crop Details :\n")
    st.write("1. '+' symbol indicates that fertilizer has to be increased.\n 2. '-' symbol indicates that fertilizer has to be decreased.\n")
    for m in dic.keys():
        if crop != m:
            k = dic[m]
            O = [f"{'+%.2f' % (i-j) if i >= j else '-%.2f' % (j - i)}" for i, j in zip(l, k)]
            m = m.capitalize()
            res[m] = O
    headers = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "Ph_value"]
    df = pd.DataFrame(res) 
    df = df.transpose() 
    df.columns = headers
    st.table(df)
    

def solve(l):
    data = pd.read_csv(r"Dataset/Crop_recommendation.csv")
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph']]
    y = data['label']
    testing_data = pd.DataFrame([l],  columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph'])
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    rf_classifier = joblib.load(rf_model_path)
    model = tf.keras.models.load_model(rnn_model_path)
    rf_prediction = rf_classifier.predict(testing_data)
    rnn_prediction = np.argmax(model.predict(np.expand_dims(testing_data, axis=2)), axis=1)
    cmb_predc = (rf_prediction + rnn_prediction) / 2
    cmb_predc_ind = cmb_predc.astype(int)
    prediction = label_encoder.inverse_transform(cmb_predc_ind)
    return prediction

def display_Contact_us():
    # st.subheader('Contact Us')
    content = 'Project Was Developed by the students of Aditya College Of Engineering & Technology of Computer Science and Engineering Department.'
    st.markdown(f"<span style='font-size: 18px;'>{content}</span>", unsafe_allow_html = True)
    designation = "M.Tech Ph.D.,"
    st.markdown(f"### Project Guide: DR. R. V. S. Lalitha <span style='font-size: 15px;'>{designation}</span>", unsafe_allow_html=True)
    names = ["J. Siva Shankar", "B. Divya Bala Tripura Sundari", "M. V. Sri Padma", "N. Siya Sudiksha"]
    emails = ["20P31A0590@acet.ac.in", "20P31A0572@acet.ac.in", "20P31A05A2@acet.ac.in", "20P31A05A6@acet.ac.in"]
    address = "Aditya College of Engineering & Technology \n\n"\
              "Surampalem, Gandepalli Mandal - 533437 \n\n" \
              "Andhra Pradesh - India."
    st.subheader("Team Members:\n")
    for name, email in zip(names, emails):
        st.write(f"**{name}:** {email}")
    st.subheader("College Address :")
    st.markdown(address, unsafe_allow_html=True)

a = st.sidebar.radio("Navigation Bar", ["Home", "Analysis", "Contact Us"])

if a == "Home":
    display_Home()
elif a == "Analysis":
    display_Analysis()        
elif a == "Contact Us":
    display_Contact_us() 
