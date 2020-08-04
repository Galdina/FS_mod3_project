import string as st

def print_cat(df, col):
    gb = df.groupby(col)["churn"].value_counts().to_frame().rename({"churn": "Number of Customers"}, axis = 1).reset_index()
    t = col[:1].upper() + col[1:]
    xlabel_rotation = 0
    if len(col)>10:
        xlabel_rotation = 90
    ax = sns.barplot(x = col, y = "Number of Customers", data = gb, hue = "churn", 
            palette = sns.color_palette("Pastel2", 8)).set_title(f"{t} and relative Churn Rates in our population")
    # ax.set_xticklabels(rotation=90)
    plt.show()
    
    


def plot_contract(df, contract_type, y_limit):
    churn = df[(df["internetservice"] != "No") & (df["phoneservice"] == "Yes") & (df["churn"] == "Yes") & (df["contract"] == contract_type)]
    nonchurn = df[(df["internetservice"] != "No") & (df["phoneservice"] == "Yes") & (df["churn"] == "No") & (df["contract"] == contract_type)]
    bins=np.arange(0,churn['tenure'].max()+3,3)    
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.figsize': [12.0, 5.0]})
    #title for two plots
    plt.suptitle(contract_type + ' contract')
    # compare two plots
    # first
    plt.subplot(1, 2, 1)
    plt.title("Former Customers")
    plt.xlabel("Tenure Length (Months)")
    plt.ylabel("Frequency")
    #set limits on the axes    
    plt.xlim(-2.5, 72)
    # y_limit - can be calculate
    plt.ylim(0, y_limit)
    # data for plot
    plt.hist(data=churn, x="tenure", bins=bins)

    #second
    plt.subplot(1, 2, 2)
    plt.title("Active Customers")
    plt.xlabel("Tenure Length (Months)")
    plt.ylabel("Frequency")
    # set limits on the axes
    plt.xlim(-2.5, 72)
    plt.ylim(0, y_limit)
    plt.hist(data=nonchurn, x="tenure", bins=bins)
    plt.show()
