import pandas as pd
import requests
from bs4 import BeautifulSoup as Soup
import pandas as pd
import os
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup as Soup
import pandas as pd
import os
from datetime import datetime, timedelta


def scraping(user_location):
    locations = ["Mumbai", "New Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad", "Ahmedabad", "Pune", "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad", "Patna", "Vadodara"]
    # Function to scrape data from Booking.com
    def scrape_bookingdotcom(destination, checkin_date, checkout_date):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
        }
        req = requests.get(
            f"https://www.booking.com/searchresults.en-gb.html?ss={destination}&checkin={checkin_date}&checkout={checkout_date}&offset==0",
            headers=headers).text
        soup = Soup(req, 'html.parser')
        ap = soup.find("ol", {"class": "a8b500abde"}).text

        df = pd.DataFrame(columns=["price", "location", "distance", "amenities", "ratings", "type"])
        for pages in range(0, int(ap[len(ap) - 1])):
            req = requests.get(
                f"https://www.booking.com/searchresults.en-gb.html?ss={destination}&checkin={checkin_date}&checkout={checkout_date}&offset=={pages * 25}",
                headers=headers).text
            soup = Soup(req, 'html.parser')
            apts = soup.find_all("div", {"class": "d20f4628d0"})
            rows = []

            for a in range(0, len(apts)):
                obj = {}

                try:
                    obj["price"] = apts[a].find("span", {"class": "fcab3ed991 fbd1d3018c e729ed5ab6"}).text
                except:
                    obj["price"] = None

                try:
                    obj["distance"] = apts[a].find("span", {"class": "cb5ebe3ffb"}).text
                except:
                    obj["distance"] = None

                try:
                    ap1 = apts[a].find('a', href=True)
                    link = ap1['href']
                    req1 = requests.get(link, headers=headers).text
                    soup2 = Soup(req1, 'html.parser')
                    obj["amenities"] = soup2.find("div", {"class": "e5e0727360"}).text
                except:
                    obj["amenities"] = None

                try:
                    obj["ratings"] = apts[a].find("div", {"class": "b5cd09854e d10a6220b4"}).text
                except:
                    obj["ratings"] = None

                try:
                    obj["type"] = apts[a].find("span", {"class": "df597226dd"}).text
                except:
                    obj["type"] = None

                try:
                    obj["location"] = apts[a].find("span", {"class": "f4bd0794db b4273d69aa"}).text
                except:
                    obj["location"] = None

                rows.append(obj)

            df = pd.concat([df, pd.DataFrame(rows)])

        # Data cleaning
        df["price"] = df["price"].str.replace(r"₹", "")
        df["price"] = df["price"].str.replace(r" ", "")
        df["price"] = df["price"].str.replace(r",", "")
        df["price"] = df["price"].str.strip()
        df['price'] = pd.to_numeric(df['price'])
        df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
        df['ratings'] = df['ratings'].fillna(df['ratings'].mean())

        return df

    # Take user input for the location, check-in date, and check-out date
    user_location=user_location.strip().capitalize()
    current_date = datetime.now().strftime("%Y-%m-%d")
    checkin_date = current_date
    checkout_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    folder_name = f"data{current_date}"
    data_folder_path = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_folder_path):
       os.makedirs(data_folder_path)
   # Step 3: Create the new folder inside the "data" folder
    new_folder_path = os.path.join(data_folder_path, folder_name)
    if not os.path.exists(new_folder_path):
       os.makedirs(new_folder_path)
    # Check if user input location has already been scraped
    user_location_csv = f"{user_location}_{current_date}.csv"
    destination_file_path = os.path.join(new_folder_path, user_location_csv)

    if os.path.isfile(destination_file_path):
        print(f"Skipping {user_location}. Already scraped.")
        user_df=pd.read_csv(destination_file_path)
    else:
        # Scrape data for the user input location
        print(f"scraping data for{user_location}\n")
        df = scrape_bookingdotcom(user_location, checkin_date, checkout_date)

        # Save the data to a CSV file with current date in the filename
        df.to_csv(destination_file_path, index=False)
        print(f"Scraped and saved data for {user_location}.")
        user_df=df
    # Scrape data for remaining locations
    for location in locations:
        location_csv = f"{location}_{current_date}.csv"
        location_file_path = os.path.join(new_folder_path, location_csv)

        if not os.path.isfile(location_file_path):
            # Scrape data for the location
            print(f"scraping data for {location}\n")

            df = scrape_bookingdotcom(location, checkin_date, checkout_date)

            # Save the data to a CSV file with current date in the filename
            df.to_csv(location_file_path, index=False)
            print(f"Scraped and saved data for {location}.")

    # Combine all CSV files into a single dataframe
    combined_df = pd.DataFrame()
    if user_location in locations:
            csv_files =[os.path.join(new_folder_path,f"{location}_{current_date}.csv") for location in locations]
    else:
        csv_files = [destination_file_path] + [os.path.join(new_folder_path,f"{location}_{current_date}.csv") for location in locations]
    print(csv_files)
    for csv_file in csv_files:
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
            combined_df = pd.concat([combined_df, df])

    # Save the combined dataframe to a CSV file with current date in the filename
    combined_folder_path = os.path.join(data_folder_path,'combined')
    if not os.path.exists(combined_folder_path):
       os.makedirs(combined_folder_path)
    final_csv_filename = f"combined_{current_date}.csv"
    file_path = os.path.join(combined_folder_path, final_csv_filename)

    combined_df.to_csv(file_path, index=False)

    print("Scraping completed.")
    return user_df



