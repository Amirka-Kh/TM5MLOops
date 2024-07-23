import gradio as gr
import json
import requests
import pandas as pd


# Define the predict function with parameters for each feature in your dataset
def predict(property_type=None, room_type=None, accommodates=None, bathrooms=None, bed_type=None,
            cancellation_policy=None, cleaning_fee=None, city=None, host_has_profile_pic=None,
            host_identity_verified=None, host_response_rate=None, instant_bookable=None,
            latitude=None, longitude=None, name=None, number_of_reviews=None,
            review_scores_rating=None, thumbnail_url=None, bedrooms=None, beds=None,
            zipcode_freq=None, neighbourhood_freq=None, detector=None, dryer=None,
            essentials=None, friendly=None, heating=None, smoke=None, tv=None,
            apartment=None, bed=None, bedroom=None, private=None, restaurants=None,
            room=None, walk=None, first_review_year=None, first_review_month=None,
            first_review_day=None, last_review_year=None, last_review_month=None,
            last_review_day=None, host_since_year=None, host_since_month=None,
            host_since_day=None, room_type_Private_room=None, room_type_Shared_room=None,
            bed_type_Couch=None, bed_type_Futon=None, bed_type_Pull_out_Sofa=None,
            bed_type_Real_Bed=None, city_Chicago=None, city_DC=None, city_LA=None,
            city_NYC=None, city_SF=None):
    # Create a dictionary for the input features
    features = {
        "property_type": property_type,
        "room_type": room_type,
        "accommodates": accommodates,
        "bathrooms": bathrooms,
        "bed_type": bed_type,
        "cancellation_policy": cancellation_policy,
        "cleaning_fee": cleaning_fee,
        "city": city,
        "host_has_profile_pic": host_has_profile_pic,
        "host_identity_verified": host_identity_verified,
        "host_response_rate": host_response_rate,
        "instant_bookable": instant_bookable,
        "latitude": latitude,
        "longitude": longitude,
        "name": name,
        "number_of_reviews": number_of_reviews,
        "review_scores_rating": review_scores_rating,
        "thumbnail_url": thumbnail_url,
        "bedrooms": bedrooms,
        "beds": beds,
        "zipcode_freq": zipcode_freq,
        "neighbourhood_freq": neighbourhood_freq,
        "detector": detector,
        "dryer": dryer,
        "essentials": essentials,
        "friendly": friendly,
        "heating": heating,
        "smoke": smoke,
        "tv": tv,
        "apartment": apartment,
        "bed": bed,
        "bedroom": bedroom,
        "private": private,
        "restaurants": restaurants,
        "room": room,
        "walk": walk,
        "first_review_year": first_review_year,
        "first_review_month": first_review_month,
        "first_review_day": first_review_day,
        "last_review_year": last_review_year,
        "last_review_month": last_review_month,
        "last_review_day": last_review_day,
        "host_since_year": host_since_year,
        "host_since_month": host_since_month,
        "host_since_day": host_since_day,
        "room_type_Private room": room_type_Private_room,
        "room_type_Shared room": room_type_Shared_room,
        "bed_type_Couch": bed_type_Couch,
        "bed_type_Futon": bed_type_Futon,
        "bed_type_Pull-out Sofa": bed_type_Pull_out_Sofa,
        "bed_type_Real Bed": bed_type_Real_Bed,
        "city_Chicago": city_Chicago,
        "city_DC": city_DC,
        "city_LA": city_LA,
        "city_NYC": city_NYC,
        "city_SF": city_SF
    }

    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])

    # Convert to JSON
    example = raw_df.iloc[0, :].to_dict()
    example = json.dumps({"inputs": example})

    # Send POST request with the payload to the deployed Model API
    response = requests.post(
        url=f"http://localhost:5151/invocations",
        data=example,
        headers={"Content-Type": "application/json"},
    )

    try:
        my_response = requests.post(
            url=f"http://localhost:5001/predict",
            data=example,
            headers={"Content-Type": "application/json"},
        )
        print(my_response.json())
    except Exception as err:
        print(err)

    # Return the model's prediction
    return response.json()


# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="property_type"),
        gr.Text(label="room_type"),
        gr.Number(label="accommodates"),
        gr.Number(label="bathrooms"),
        gr.Text(label="bed_type"),
        gr.Number(label="cancellation_policy"),
        gr.Number(label="cleaning_fee"),
        gr.Text(label="city"),
        gr.Number(label="host_has_profile_pic"),
        gr.Number(label="host_identity_verified"),
        gr.Number(label="host_response_rate"),
        gr.Number(label="instant_bookable"),
        gr.Number(label="latitude"),
        gr.Number(label="longitude"),
        gr.Text(label="name"),
        gr.Number(label="number_of_reviews"),
        gr.Number(label="review_scores_rating"),
        gr.Text(label="thumbnail_url"),
        gr.Number(label="bedrooms"),
        gr.Number(label="beds"),
        gr.Number(label="zipcode_freq"),
        gr.Number(label="neighbourhood_freq"),
        gr.Number(label="detector"),
        gr.Number(label="dryer"),
        gr.Number(label="essentials"),
        gr.Number(label="friendly"),
        gr.Number(label="heating"),
        gr.Number(label="smoke"),
        gr.Number(label="tv"),
        gr.Number(label="apartment"),
        gr.Number(label="bed"),
        gr.Number(label="bedroom"),
        gr.Number(label="private"),
        gr.Number(label="restaurants"),
        gr.Number(label="room"),
        gr.Number(label="walk"),
        gr.Number(label="first_review_year"),
        gr.Number(label="first_review_month"),
        gr.Number(label="first_review_day"),
        gr.Number(label="last_review_year"),
        gr.Number(label="last_review_month"),
        gr.Number(label="last_review_day"),
        gr.Number(label="host_since_year"),
        gr.Number(label="host_since_month"),
        gr.Number(label="host_since_day"),
        gr.Number(label="room_type_Private room"),
        gr.Number(label="room_type_Shared room"),
        gr.Number(label="bed_type_Couch"),
        gr.Number(label="bed_type_Futon"),
        gr.Number(label="bed_type_Pull-out Sofa"),
        gr.Number(label="bed_type_Real Bed"),
        gr.Number(label="city_Chicago"),
        gr.Number(label="city_DC"),
        gr.Number(label="city_LA"),
        gr.Number(label="city_NYC"),
        gr.Number(label="city_SF")
    ],
    outputs=gr.Text(label="prediction result"),
    examples="data/examples"
)

# Launch the web UI locally on port 5155
demo.launch(server_port=5155)

# Launch the web UI in Gradio cloud on port 5155
# demo.launch(share=True, server_port=5155)
