from tensorflow.keras import layers, models
import numpy as np
import csv
import random
import sys
import os


class PeopleSet:
    def __init__(self):
        self.ids = []
        self.labels = []
        self.features = []
        self.norm = []

    @staticmethod
    def Normalize_samples(data):
        PeopleSet.mean = data.mean(axis=0)
        data -= PeopleSet.mean
        PeopleSet.std = data.std(axis=0)
        return data / PeopleSet.std

    def parse_csv(self, file_name):
        self.lex_dict = {"male": "0", "female": "1",
                         "C": "1", "S": "2", "Q": "3"}

        with open(file_name, 'r') as csvfile:
            spamreader = csv.DictReader(csvfile)
            for row in spamreader:

                self.labels.append(row['Survived'])
                self.ids.append(row['PassengerId'])
                if (row['Age'] == ""):
                    age = 29.7

                else:
                    age = row['Age']

                if (row['Fare'] == ""):
                    fare = 0

                else:
                    fare = row["Fare"]

                if (row['Embarked'] == ""):
                    embark = random.randint(1, 3)

                else:
                    embark = self.lex_dict[row['Embarked']]

                self.features.append(
                                    np.array(
                                            [
                                                row['Pclass'],
                                                self.lex_dict[row['Sex']],
                                                age, row['SibSp'],
                                                row['Parch'], fare,
                                                embark]).astype(float))

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)


def build_model(train):
    model = models.Sequential()
    model.add(layers.Dense(
            256, activation='relu',
            input_shape=(train.shape[1],)))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return model


def k_fold(norm_data, labels):
    k = 4
    num_val_samples = len(norm_data) // k
    num_epochs = 100
    all_scores = []

    for i in range(k):

        val_data = norm_data[i * num_val_samples:
                             (i + 1) * num_val_samples]
        val_targets = labels[i * num_val_samples:
                             (i + 1) * num_val_samples]

        partial_norm_data = np.concatenate([
                                            norm_data[:i * num_val_samples],
                                            norm_data[(i + 1) *
                                                      num_val_samples:]],
                                           axis=0)

        partial_labels = np.concatenate(
            [labels[:i * num_val_samples],
             labels[(i + 1) * num_val_samples:]],
            axis=0)

        model = build_model(norm_data)

        model.fit(partial_norm_data, partial_labels,
                  epochs=num_epochs, batch_size=1, verbose=1)

        val_loss, val_acc = model.evaluate(val_data,
                                           val_targets,
                                           verbose=0)
        all_scores.append(val_acc)


def main():
    train_ps = PeopleSet()
    train_ps.parse_csv(sys.argv[1])

    model = build_model(PeopleSet.Normalize_samples(train_ps.features))

    model.fit(
        PeopleSet.Normalize_samples(train_ps.features),
        train_ps.labels, epochs=100, batch_size=16, verbose=1)

    test_labels = []
    test_input = []
    with open('test.csv') as f:
        for lines in f.readlines()[1:]:
            lines = lines.split(',')

            curr_data = []
            curr_data.append(lines[1])
            curr_data.append(train_ps.lex_dict[lines[4]])

            if(lines[5] == ""):
                curr_data.append(29.6)
            else:
                curr_data.append(lines[5])

            curr_data.append(lines[6])
            curr_data.append(lines[7])

            if(lines[9] == ""):
                curr_data.append(random.randint(1, 3))

            else:
                curr_data.append(lines[9])

            if(lines[11] == ""):
                curr_data.append(random.randint(1, 3))
            else:
                curr_data.append(
                                    train_ps.lex_dict[
                                                        lines[11]
                                                        .replace('\n', '')
                                                        .replace('\r', '')])

            test_labels.append(lines[0])
            test_input.append(np.array(curr_data, dtype=float))

    print()
    test_input -= PeopleSet.mean
    test_input /= PeopleSet.std

    pred = 0
    model_pred = model.predict(test_input)
    print("PassengerId,Survived")
    for i in range(len(test_labels)):
        if model_pred[i][0] < 0.75:
            pred = 0
        else:
            pred = 1
        print(str(test_labels[i])+","+str(pred))

main()
