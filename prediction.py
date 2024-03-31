import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def read_csv_files():
    # Read the age demographics data
    demographics = pd.read_csv('ages.txt', sep='\t')

    # Read the age opinions data
    opinions = pd.read_csv('ageopinions.txt', sep='\t')

    return demographics, opinions

def read_party_opinions():
    # Read the party opinions data
    opinions = pd.read_csv('partyopinions.txt', sep='\t')

    # Convert the DataFrame to a dictionary
    opinions_dict = opinions.set_index('Puolue').T.to_dict()

    return opinions_dict

def plot_opinion_values_with_fit(opinions, show_data=True, show_fit=True, plot = True):
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Create a dictionary to map age groups to the middle number of each range
    age_groups = opinions['Ikä'].unique()
    age_group_dict = {age_group: (float(age_group.split('-')[0]) + float(age_group.split('-')[1])) / 2 for age_group in age_groups}

    # Replace the age groups with the middle number of each range for the purpose of fitting the polynomial
    x = opinions['Ikä'].map(age_group_dict)
    
    # Initialize a dictionary to store the polynomial coefficients
    poly_coeffs = {}

    # Plot the opinion values for each age group and fit a 4th order polynomial
    for column in opinions.columns[1:]:
        y = opinions[column]
        if show_data:
            ax.plot(x, y, label=column)

        if show_fit:
            # Fit a 4th order polynomial
            poly = np.polyfit(x, y, 4)
            poly_y = np.poly1d(poly)

            # Store the polynomial coefficients
            poly_coeffs[column] = poly

            # Create a new, denser array of x-values
            x_dense = np.linspace(min(x), max(x), 500)

            # Plot the polynomial
            ax.plot(x_dense, poly_y(x_dense), label=f'{column} fit')

    # Set the x-ticks to be the age group strings
    ax.set_xticks(list(age_group_dict.values()))
    ax.set_xticklabels(list(age_group_dict.keys()))

    # Set the labels and title
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Opinion Value')
    ax.set_title('Opinion Values by Age Group with Polynomial Fit')

    # Add a legend
    ax.legend()

    # Show the plot
    if plot:
        plt.show()

    # Return the polynomial coefficients
    return poly_coeffs


def predict_party_opinions(demographics, poly_coeffs):
    # Create a dictionary to map age groups to the middle number of each range
    age_groups = demographics['Ikä'].unique()
    age_group_dict = {age_group: (float(age_group.split('-')[0]) + float(age_group.split('-')[1])) / 2 for age_group in age_groups}
    age_group_list = []
    for key in age_group_dict:
        age_group_list.append(age_group_dict[key])
    if debug: print(f"ages: {age_group_dict}\n")
    
    demographics.set_index('Ikä', inplace=True)

    # Calculate the predicted opinions for each party
    parties = ['VAS', 'SDP', 'VIHR', 'KESK', 'RKP', 'KD', 'KOK', 'PS']
    opinion_values = ["Täysin_samaa",	"jokseenkin_samaa",	"neutral",	"eos",	"jokseenkin_eri",	"täysin_eri"]
    predictions = {}
    for party in parties:
        predictions[party] = {}
        for opinion in opinion_values:
            predictions[party][opinion] = 0
            # Use the polynomial coefficients to calculate the predicted opinions
            poly = np.poly1d(poly_coeffs[opinion])
            for age in age_group_dict:
                predictions[party][opinion] += poly(age_group_dict[age])*demographics[party][age]

    # Normalize the predictions so that they sum up to 100 for each party
    for party in parties:
        total = sum(predictions[party].values())
        for opinion in opinion_values:
            predictions[party][opinion] = (predictions[party][opinion] / total) * 100

    return predictions


def plot_predictions(predictions):
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Get the parties and opinions
    parties = list(predictions.keys())
    opinions = list(predictions[parties[0]].keys())

    # Create an array for the x-values
    x = np.arange(len(parties))

    # Create a color map
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])

    # Create a stacked bar for each opinion
    for i, opinion in enumerate(opinions):
        opinion_values = [predictions[party][opinion] for party in parties]
        if i == 0:
            bars = ax.bar(x, opinion_values, label=opinion, color=cmap(i/len(opinions)))
            bottom_values = opinion_values
        else:
            bars = ax.bar(x, opinion_values, bottom=bottom_values, label=opinion, color=cmap(i/len(opinions)))
            bottom_values = [sum(x) for x in zip(bottom_values, opinion_values)]

        # Write the percentage values on the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2,
                    f'{height:.1f}%', ha='center', va='center', color='white')

    # Set the x-ticks to be the party names
    ax.set_xticks(x)
    ax.set_xticklabels(parties)

    # Set the labels and title
    ax.set_xlabel('Party')
    ax.set_ylabel('Predicted Opinion (%)')
    ax.set_title('Predicted Party Opinions')

    # Add a legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_predictions_difference(predictions1, predictions2):
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Get the parties and opinions
    parties = list(predictions1.keys())
    opinions = list(predictions1[parties[0]].keys())

    # Create an array for the x-values
    x = np.arange(len(parties))

    # Create a color map
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])

    # Create a stacked bar for each opinion
    for i, opinion in enumerate(opinions):
        opinion_values1 = [predictions1[party][opinion] for party in parties]
        opinion_values2 = [predictions2[party][opinion] for party in parties]
        if i == 0:
            bars = ax.bar(x, opinion_values1, label=opinion, color=cmap(i/len(opinions)))
            bottom_values = opinion_values1
        else:
            bars = ax.bar(x, opinion_values1, bottom=bottom_values, label=opinion, color=cmap(i/len(opinions)))
            bottom_values = [sum(x) for x in zip(bottom_values, opinion_values1)]

        # Write the percentage values and the percentage difference on the bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2,
                    f'{height:.1f}%\n({opinion_values1[j] - opinion_values2[j]:.1f}%)',
                    ha='center', va='center', color='white')

    # Set the x-ticks to be the party names
    ax.set_xticks(x)
    ax.set_xticklabels(parties)

    # Set the labels and title
    ax.set_xlabel('Party')
    ax.set_ylabel('Predicted Opinion (%)')
    ax.set_title('Predicted Party Opinions')

    # Add a legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()

debug = 1

demographics, opinions = read_csv_files()
print(f"Demographics: \n{demographics}\n")
coeffs = plot_opinion_values_with_fit(opinions, plot = False)
if debug: print(f"\nCoeffs: {coeffs}\n")


opinions = read_party_opinions()
if debug: print(f"\nOpinions: {opinions}\n")
predictions = predict_party_opinions(demographics=demographics, poly_coeffs=coeffs)
if debug: print(f"\nPredictions: {predictions}\n")

plot_predictions(predictions)
plot_predictions_difference(predictions, opinions)