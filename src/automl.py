import numpy as np
import openml

from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score


datasets_all = [
    "iris", "digits", "wine", "breastcancer", "analcatdataauthorship",
    "bloodtransfusion", "monks1", "monks2", "steelplatesfault",
    "qsarbiodeg", "phoneme", "diabetes", "hillvalley", "eegeyestate",
    "waveform", "spambase", "australian", "churn", "vehicle",
    "balancescale", "kc1", "kc2", "cardiotocography",
    "wallrobotnavigation", "segment", "artificialcharacters",
    "electricity", "gasdrift", "olivetti", "letter"
]

dict_models = {
    0: "adaboost", 1: "gradientboosting", 2: "decisiontree",
    3: "extratrees", 4: "randomforest"
}

# category: 0-4, simplex
# n_estimtaors: 5, int
# max_depth: 6, int
# max_features: 7, float
# learning_rate: 8, float, log

bounds = np.array([
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [10, 200],
    [2, 8],
    [0.2, 1.0],
    [-3.0, 1.0],
])


def get_train_test_datasets(dataset, seed=42):
    print("dataset: {} seed: {}".format(dataset, seed))

    if dataset == "iris":
        ds = datasets.load_iris()
        X = ds.data
        y = ds.target
    elif dataset == "digits":
        ds = datasets.load_digits()
        X = ds.data
        y = ds.target
    elif dataset == "wine":
        ds = datasets.load_wine()
        X = ds.data
        y = ds.target
    elif dataset == "breastcancer":
        ds = datasets.load_breast_cancer()
        X = ds.data
        y = ds.target
    elif dataset == "olivetti":
        ds = fetch_olivetti_faces()
        # get data point size
        img_rows, img_cols = ds.images.shape[1:]
        dim = img_rows * img_cols
        X = ds.data
        X = X.reshape(-1, dim)
        y = ds.target
    elif dataset == "analcatdataauthorship":
        dataset_id = 458
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "Austen"] = "0"
        y_new[y_new == "London"] = "1"
        y_new[y_new == "Milton"] = "2"
        y_new[y_new == "Shakespeare"] = "3"
        y = np.array([int(val) for val in y_new])
    elif dataset == "bloodtransfusion":
        dataset_id = 1464
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "monks1":
        dataset_id = 333
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "monks2":
        dataset_id = 334
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "steelplatesfault":
        dataset_id = 1504
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "qsarbiodeg":
        dataset_id = 1494
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "phoneme":
        dataset_id = 1489
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "diabetes":
        dataset_id = 37
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "tested_negative"] = "0"
        y_new[y_new == "tested_positive"] = "1"
        y = np.array([int(val) for val in y_new])
    elif dataset == "hillvalley":
        dataset_id = 1479
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "eegeyestate":
        dataset_id = 1471
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "kc1":
        dataset_id = 1067
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([str(val) for val in y.values])
        y_new[y_new == "False"] = "0"
        y_new[y_new == "True"] = "1"
        y = np.array([int(val) for val in y_new])
    elif dataset == "kc2":
        dataset_id = 1063
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "no"] = "0"
        y_new[y_new == "yes"] = "1"
        y = np.array([int(val) for val in y_new])
    elif dataset == "spambase":
        dataset_id = 44
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "electricity":
        dataset_id = 151
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "UP"] = "0"
        y_new[y_new == "DOWN"] = "1"
        y = np.array([int(val) for val in y_new])
    elif dataset == "australian":
        dataset_id = 40981
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "letter":
        dataset_id = 6
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        # convert letters to numbers
        y = np.array([ord(val) for val in y.values])
    elif dataset == "churn":
        dataset_id = 40701
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "vehicle":
        dataset_id = 54
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "opel"] = "0"
        y_new[y_new == "saab"] = "1"
        y_new[y_new == "bus"] = "2"
        y_new[y_new == "van"] = "3"
        y = np.array([int(val) for val in y_new])
    elif dataset == "balancescale":
        dataset_id = 11
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "L"] = "0"
        y_new[y_new == "B"] = "1"
        y_new[y_new == "R"] = "2"
        y = np.array([int(val) for val in y_new])
    elif dataset == "artificialcharacters":
        dataset_id = 1459
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "cardiotocography":
        dataset_id = 1466
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "wallrobotnavigation":
        dataset_id = 1497
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "waveform":
        dataset_id = 60
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "gasdrift":
        dataset_id = 1476
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "connect4":
        dataset_id = 40668
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    elif dataset == "segment":
        dataset_id = 40984
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "brickface"] = "0"
        y_new[y_new == "sky"] = "1"
        y_new[y_new == "foliage"] = "2"
        y_new[y_new == "cement"] = "3"
        y_new[y_new == "window"] = "4"
        y_new[y_new == "path"] = "5"
        y_new[y_new == "grass"] = "6"
        y = np.array([int(val) for val in y_new])
    else:
        raise ValueError('')

    X = MinMaxScaler().fit_transform(X)
    y = np.array([int(val) for val in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    return X_train, y_train, X_test, y_test

def train_test_adaboost(params, train_X, train_y, test_X, test_y, seed):
    model_name = "Adaboost"
    print(model_name, params, seed)

    model = AdaBoostClassifier(
        n_estimators=int(params[5]),
        learning_rate=10**(float(params[8])),
        random_state=seed
    )

    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    auc = accuracy_score(test_y, pred_y)

    loss = 1.0 - auc
    del model

    return loss

def train_test_gradientboosting(params, train_X, train_y, test_X, test_y, seed):
    model_name = "GradientBoosting"
    print(model_name, params, seed)

    model = GradientBoostingClassifier(
        n_estimators=int(params[5]),
        max_depth=int(params[6]),
        max_features=float(params[7]),
        learning_rate=10**(float(params[8])),
        random_state=seed
    )
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    auc = accuracy_score(test_y, pred_y)

    loss = 1.0 - auc
    del model

    return loss

def train_test_decisiontree(params, train_X, train_y, test_X, test_y, seed):
    model_name = "DecisionTree"
    print(model_name, params, seed)

    model = DecisionTreeClassifier(
        max_depth=int(params[6]),
        max_features=float(params[7]),
        random_state=seed
    )

    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    auc = accuracy_score(test_y, pred_y)

    loss = 1.0 - auc
    del model

    return loss

def train_test_extratrees(params, train_X, train_y, test_X, test_y, seed):
    model_name = "ExtraTrees"
    print(model_name, params, seed)

    model = ExtraTreesClassifier(
        n_estimators=int(params[5]),
        max_depth=int(params[6]),
        max_features=float(params[7]),
        random_state=seed
    )

    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    auc = accuracy_score(test_y, pred_y)

    loss = 1.0 - auc
    del model

    return loss

def train_test_randomforest(params, train_X, train_y, test_X, test_y, seed):
    model_name = "RandomForest"
    print(model_name, params, seed)

    model = RandomForestClassifier(
        n_estimators=int(params[5]),
        max_depth=int(params[6]),
        max_features=float(params[7]),
        random_state=seed
    )

    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    auc = accuracy_score(test_y, pred_y)

    loss = 1.0 - auc
    del model

    return loss

def fun_target(params, train_X, train_y, test_X, test_y, seed):
    model = params[0:5]
    model = dict_models[np.argmax(model)]

    if model == "adaboost":
        res = train_test_adaboost(params, train_X, train_y, test_X, test_y, seed)
    elif model == "gradientboosting":
        res = train_test_gradientboosting(params, train_X, train_y, test_X, test_y, seed)
    elif model == "decisiontree":
        res = train_test_decisiontree(params, train_X, train_y, test_X, test_y, seed)
    elif model == "extratrees":
        res = train_test_extratrees(params, train_X, train_y, test_X, test_y, seed)
    elif model == "randomforest":
        res = train_test_randomforest(params, train_X, train_y, test_X, test_y, seed)
    else:
        raise ValueError('')

    print(res)

    return res
