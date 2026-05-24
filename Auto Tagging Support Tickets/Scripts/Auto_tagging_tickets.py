import pandas as pd

# Load Kaggle dataset
df=pd.read_csv("D:\Pprojects\MLintern_II\Auto Tagging Support Tickets\Data\customer_support_tickets.csv")
print(df.info())
# print(df)
# Focus on ticket description and ticket type
df = df[["Ticket Type", "Ticket Description"]]

# Strip spaces from all column names
df.columns = df.columns.str.strip()

# Rename specific columns
df = df.rename(columns={
    "Ticket Type": "label",
    "Ticket Description": "text"
})
print(df.info())
print(df.describe())



