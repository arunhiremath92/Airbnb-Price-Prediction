import pandas as pd

df = pd.read_csv("listings_summary.csv", sep=',', skipinitialspace=True);


def printCols(df):
    for col in df.columns:
        print(col)

# Converts amenities to amenities count
def count_amenities(df):
    pd.options.mode.chained_assignment = None
    length = len(df["amenities"])
    for i in range(0, length):
        value = (df["amenities"][i]).split(",")
        df["amenities"][i] = len(value)

# remove $ sign in extra people column price amount
def clean_extra_people_price(df):
    length = len(df["extra_people"])
    for i in range(0, length):
        df["extra_people"][i] = (df["extra_people"][i]).replace("$", "")


features_having_boolean = ["require_guest_phone_verification", "host_is_superhost", "host_has_profile_pic",
                           "host_identity_verified", "is_location_exact", "requires_license", "instant_bookable"]


def boolean_to_numeric(data):
    if data == "t" or data == "T":
        return 1.0
    elif data == "f" or data == "F":
        return 0.0
    else:
        return None

# Convert the boolean columns to numeric values
def convert_boolean_features(df):
    for i in features_having_boolean:
         df[i] = df[i].map(boolean_to_numeric)



# print(len(df["amenities"][0].split(",")))

count_amenities(df)
clean_extra_people_price(df)
print(df["require_guest_phone_verification"].head())
convert_boolean_features(df)
# print(df["amenities"].value_counts())

print("#############")
#print(df["require_guest_phone_verification"].head())
# print(len(df["amenities"].value_counts()))

#for col in df["require_guest_phone_verification"]:
#   print(col)
