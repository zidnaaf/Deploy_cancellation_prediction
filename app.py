import numpy as np
import pickle
import streamlit as st
install imblearn
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.pipeline import Pipeline, make_pipeline

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb')) # Reading the binary format


# Creating a function for prediction

def cancellation_prediction(input_data):
    
    # change the input data to numpy array
    input_data_np = np.asarray(input_data)

    # Reshape the array
    input_data_reshape = input_data_np.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0] == 0):
        return 'The Booking order is cancelled'
    else:
        return 'The Booking order is not cancelled'


def main():

    # Give title
    st.title('Booking Cancellation Prediction')

    # Get the input data

    no_of_adults = st.text_input('Number of Adults')
    no_of_children = st.text_input('Number of children')
    no_of_weekend_nights = st.text_input('How many night on weekend')
    no_of_week_nights =	st.text_input('How many night on weekday')
    required_car_parking_space = st.text_input('Using the parking space (Yes=1;No=0)')
    lead_time =	st.text_input('The Booking Lead Time')
    arrival_year = st.text_input('Year of arrival date')
    repeated_guest = st.text_input('Is the customer a repeated guest?  (Yes=1;No=0)')
    no_of_previous_cancellations =	st.text_input('Number of previous bookings that were canceled')
    no_of_previous_bookings_not_canceled =	st.text_input('Number of previous bookings that were not canceled')
    avg_price_per_room = st.text_input('Average price per day of the reservation (in euro)')
    no_of_special_requests = st.text_input('Total number of special requests')
    ToM_Plan_1 = st.text_input('Is the customer booked meal type 1? (Yes=1;No=0)')
    ToM_Plan_2 = st.text_input('Is the customer booked meal type 2? (Yes=1;No=0)')
    ToM_Plan_3 = st.text_input('Is the customer booked meal type 3? (Yes=1;No=0)')
    ToM_Plan_NS = st.text_input('Is the customer not booked any meal type? (Yes=1;No=0)')
    Room_Type_1 = st.text_input('Is the customer booked room type number 1? (Yes=1;No=0)')
    Room_Type_2 = st.text_input('Is the customer booked room type number 2? (Yes=1;No=0)')
    Room_Type_3 = st.text_input('Is the customer booked room type number 3? (Yes=1;No=0)')
    Room_Type_4 = st.text_input('Is the customer booked room type number 4? (Yes=1;No=0)')
    Room_Type_5 = st.text_input('Is the customer booked room type number 5? (Yes=1;No=0)')
    Room_Type_6 = st.text_input('Is the customer booked room type number 6? (Yes=1;No=0)')
    Room_Type_7 = st.text_input('Is the customer booked room type number 7? (Yes=1;No=0)')
    arrival_month_1	= st.text_input('Coming in January (Yes=1;No=0)')
    arrival_month_10 = st.text_input('Coming in October (Yes=1;No=0)')
    arrival_month_11 = st.text_input('Coming in November (Yes=1;No=0)')
    arrival_month_12 = st.text_input('Coming in December (Yes=1;No=0)')
    arrival_month_2 = st.text_input('Coming in February (Yes=1;No=0)')
    arrival_month_3 = st.text_input('Coming in March (Yes=1;No=0)')
    arrival_month_4 = st.text_input('Coming in April (Yes=1;No=0)')
    arrival_month_5 = st.text_input('Coming in May (Yes=1;No=0)')
    arrival_month_6 = st.text_input('Coming in June (Yes=1;No=0)')
    arrival_month_7 = st.text_input('Coming in July (Yes=1;No=0)')
    arrival_month_8 = st.text_input('Coming in August (Yes=1;No=0)')
    arrival_month_9 = st.text_input('Coming in September (Yes=1;No=0)')
    market_segment_type_Aviation = st.text_input('Aviation market segment? (Yes=1;No=0)')
    market_segment_type_Complementary = st.text_input('Complementary market segment? (Yes=1;No=0)')
    market_segment_type_Corporate = st.text_input('Corporate market segment? (Yes=1;No=0)')
    market_segment_type_Offline	= st.text_input('Offline market segment? (Yes=1;No=0)')
    market_segment_type_Online = st.text_input('Online market segment? (Yes=1;No=0)')

    # Code for prediction
    cancellation = ''

    # Create a button for prediction
    if st.button('Booking Status'):
        cancellation = cancellation_prediction([no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
                                                required_car_parking_space, lead_time, arrival_year, repeated_guest, 
                                                no_of_previous_cancellations, no_of_previous_bookings_not_canceled, 
                                                avg_price_per_room, no_of_special_requests, ToM_Plan_1,	ToM_Plan_2,	ToM_Plan_3,
                                                ToM_Plan_NS, Room_Type_1, Room_Type_2, Room_Type_3,	Room_Type_4, Room_Type_5, 
                                                Room_Type_6, Room_Type_7, arrival_month_1, arrival_month_10, arrival_month_11,
                                                arrival_month_12, arrival_month_2, arrival_month_3, arrival_month_4, arrival_month_5,
                                                arrival_month_6, arrival_month_7, arrival_month_8, arrival_month_9, market_segment_type_Aviation,
                                                market_segment_type_Complementary, market_segment_type_Corporate, market_segment_type_Offline, market_segment_type_Online])

    st.success(cancellation)


if __name__ == '__main__':
    main()
