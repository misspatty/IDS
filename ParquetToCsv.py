import pandas

datasets = pandas.read_parquet('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/cic-collection.parquet')

# Write the DataFrame to a CSV file
datasets.to_csv('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/CIC_Collection.csv')

# Calculate the quarter points to split the dataset into four parts
# Calculate the eighth points to split the dataset into eight parts
EighthSplit1 = len(datasets) // 8
EighthSplit2 = EighthSplit1 * 2
EighthSplit3 = EighthSplit1 * 3
EighthSplit4 = EighthSplit1 * 4
EighthSplit5 = EighthSplit1 * 5
EighthSplit6 = EighthSplit1 * 6
EighthSplit7 = EighthSplit1 * 7

# Split the dataset into eight parts
part1 = datasets.iloc[:EighthSplit1]
part2 = datasets.iloc[EighthSplit1:EighthSplit2]
part3 = datasets.iloc[EighthSplit2:EighthSplit3]
part4 = datasets.iloc[EighthSplit3:EighthSplit4]
part5 = datasets.iloc[EighthSplit4:EighthSplit5]
part6 = datasets.iloc[EighthSplit5:EighthSplit6]
part7 = datasets.iloc[EighthSplit6:EighthSplit7]
part8 = datasets.iloc[EighthSplit7:]

# Write the eight parts to separate CSV files
part1.to_csv('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/part1.csv', index=False)
part2.to_csv('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/part2.csv', index=False)
part3.to_csv('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/part3.csv', index=False)
part4.to_csv('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/part4.csv', index=False)
part5.to_csv('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/part5.csv', index=False)
part6.to_csv('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/part6.csv', index=False)
part7.to_csv('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/part7.csv', index=False)
part8.to_csv('C:/Users/fatim/OneDrive/Documents/Uni/YR4/CM4105 Honours Project/Project/part8.csv', index=False)