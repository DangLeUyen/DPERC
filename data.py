from utils import *

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text), sep=' ', header=None)
    column_names = [
            "checking_account_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_account", "employment_status", "installment_rate", "personal_status",
            "other_debtors", "residence_since", "property", "age", "other_installment_plans",
            "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker","class"]

    data.columns = column_names
    data = data.dropna()
    X = data.drop(columns=["class"]).select_dtypes(include=['int64']).values
    Z = data.select_dtypes(exclude=['int64']).values
    y = data["class"].values
    X = normalize_data(X)
    label_encoder = LabelEncoder()
    for i in range(Z.shape[1]):
        Z[:,i] = label_encoder.fit_transform(Z[:,i])
    y = label_encoder.fit_transform(y)
    print("Continuous", X.shape)
    print("Class", y.shape)
    print("Categorical",  Z.shape)
    for i in np.unique(y):
        print(sum(y==i))
    print(data.shape)

    return (X, y, Z)


