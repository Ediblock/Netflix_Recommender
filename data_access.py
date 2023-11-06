import os
import numpy as np
import pandas as pd
import tensorflow as tf
import ast
import copy


class MovieDataset:
    def __init__(self, data_set: tf.data.Dataset, number_of_movies: int, highest_score=5.0, batch_size=32,
                 weight_sample: bool = False, hot_encoding: bool = False, split_test_validation: bool = True):
        self.number_of_movies = number_of_movies
        self.dataset = data_set
        self.unique_ids = []
        self.highest_score = highest_score
        self.hot_encoding = hot_encoding
        self.movie_rating_ser = pd.Series(name="movie_rating", dtype=np.float32)
        self.__get_data_from_training()
        self.batch_size = batch_size
        self.index_list = np.array(range(self.__len__()))
        self.weight_sample = weight_sample
        self._test_index_list = []
        self._validation_index_list = []
        self._train_index_list = []
        self._test_percentage_split = 10
        self._validation_percentage_split = 10
        self._split_test_validation = split_test_validation
        if self._split_test_validation:
            self.split_train_test_validation_data()
        self.__split_data_retrieve_dict = {"train": 1,
                                           "test": 2,
                                           "validation": 3}

    def set_data_test_percentage(self, test_percentage):
        self._test_percentage_split = test_percentage
        if self._split_test_validation:
            self.split_train_test_validation_data()

    def set_data_validation_percentage(self, validation_percentage):
        self._validation_percentage_split = validation_percentage
        if self._split_test_validation:
            self.split_train_test_validation_data()

    def set_hot_encoding(self, hot_encoding):
        self.hot_encoding = hot_encoding

    def get_test_length(self):
        return len(self._test_index_list)

    def get_validation_length(self):
        return len(self._validation_index_list)

    def get_train_length(self):
        return len(self._train_index_list)

    def shuffle_train_data(self):
        np.random.shuffle(self._train_index_list)

    def shuffle_test_data(self):
        np.random.shuffle(self._test_index_list)

    def shuffle_validation_data(self):
        np.random.shuffle(self._validation_index_list)

    def shuffle_all_data(self):
        self.split_train_test_validation_data(shuffle=True)

    def get_train_data(self, index):
        if self.hot_encoding:
            x1, x2, y = [], [], []
            if isinstance(index, int):
                x1, x2, y = self.get_batch_data(index, type_of_data=self.__split_data_retrieve_dict.get("train"))
            elif isinstance(index, slice):
                for i in range(*index.indices(len(self))):
                    tmp_list = self.get_batch_data(i, type_of_data=self.__split_data_retrieve_dict.get("train"))
                    x1.append(tmp_list[0])
                    x2.append(tmp_list[1])
                    y.append(tmp_list[2])
            return x1, x2, y
        else:
            x, y = None, None
            if isinstance(index, int):
                x = self.get_batch_data(index, type_of_data=self.__split_data_retrieve_dict.get("train"))
                y = x
            elif isinstance(index, slice):
                x = []
                for i in range(*index.indices(len(self))):
                    x += self.get_batch_data(i, type_of_data=self.__split_data_retrieve_dict.get("train"))
                y = x
            if not self.weight_sample:
                return x, y
            else:
                weights = self.__get_weight_sampling(x)
                return x, y, weights

    def get_test_data(self, index):
        if self.hot_encoding:
            x1, x2, y = [], [], []
            if isinstance(index, int):
                x1, x2, y = self.get_batch_data(index, type_of_data=self.__split_data_retrieve_dict.get("test"))
            elif isinstance(index, slice):
                for i in range(*index.indices(len(self))):
                    tmp_list = self.get_batch_data(i, type_of_data=self.__split_data_retrieve_dict.get("test"))
                    x1.append(tmp_list[0])
                    x2.append(tmp_list[1])
                    y.append(tmp_list[2])
            return x1, x2, y
        else:
            x, y = None, None
            if isinstance(index, int):
                x = self.get_batch_data(index, type_of_data=self.__split_data_retrieve_dict.get("test"))
                y = x
            elif isinstance(index, slice):
                x = []
                for i in range(*index.indices(len(self))):
                    x += self.get_batch_data(i, type_of_data=self.__split_data_retrieve_dict.get("test"))
                y = x
            if not self.weight_sample:
                return x, y
            else:
                weights = self.__get_weight_sampling(x)
                return x, y, weights

    def get_validation_data(self, index):
        if self.hot_encoding:
            x1, x2, y = [], [], []
            if isinstance(index, int):
                x1, x2, y = self.get_batch_data(index, type_of_data=self.__split_data_retrieve_dict.get("validation"))
            elif isinstance(index, slice):
                for i in range(*index.indices(len(self))):
                    tmp_list = self.get_batch_data(i, type_of_data=self.__split_data_retrieve_dict.get("validation"))
                    x1.append(tmp_list[0])
                    x2.append(tmp_list[1])
                    y.append(tmp_list[2])
            return x1, x2, y
        else:
            x, y = None, None
            if isinstance(index, int):
                x = self.get_batch_data(index, type_of_data=self.__split_data_retrieve_dict.get("validation"))
                y = x
            elif isinstance(index, slice):
                x = []
                for i in range(*index.indices(len(self))):
                    x += self.get_batch_data(i, type_of_data=self.__split_data_retrieve_dict.get("validation"))
                y = x
            if not self.weight_sample:
                return x, y
            else:
                weights = self.__get_weight_sampling(x)
                return x, y, weights

    def get_user_id(self, index, type_of_data: int = 0):
        user_idx = []
        length_check = False
        if type_of_data == self.__split_data_retrieve_dict.get("train"):
            index_type_data = self._train_index_list[index]
            if index_type_data < self.get_train_length():
                length_check = True
        elif type_of_data == self.__split_data_retrieve_dict.get("test"):
            index_type_data = self._test_index_list[index]
            if index_type_data < self.get_test_length():
                length_check = True
        elif type_of_data == self.__split_data_retrieve_dict.get("validation"):
            index_type_data = self._validation_index_list[index]
            if index_type_data < self.get_validation_length():
                length_check = True
        else:
            index_type_data = index

        if length_check:
            for i in range(self.batch_size):
                user_idx.append(self.movie_rating_ser.index[index * self.batch_size + i])
        else:
            for i in range(self.batch_size):
                try:
                    user_idx.append(self.movie_rating_ser.index[index * self.batch_size + i])
                except:
                    break
        return user_idx

    def split_train_test_validation_data(self, shuffle: bool = True):
        test_len, validation_len = 0, 0
        if shuffle:
            self.shuffle()
        index_list_length = len(self.index_list)
        if self._test_percentage_split:
            test_len = int(np.floor(index_list_length * self._test_percentage_split / 100.))
        if self._validation_percentage_split:
            validation_len = int(np.floor(index_list_length * self._validation_percentage_split / 100.))
        self._test_index_list = self.index_list[0:test_len]
        self._validation_index_list = self.index_list[test_len:validation_len + test_len]
        self._train_index_list = self.index_list[test_len + validation_len:]

    def set_split_test_validation_bool(self, split_test_validation: bool):
        self._split_test_validation = split_test_validation
        if self._split_test_validation:
            self.split_train_test_validation_data()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.index_list = np.array(range(self.__len__()))

    def set_weight_sample(self, weight_sample):
        self.weight_sample = weight_sample

    def __len__(self):
        return int(np.ceil(len(self.movie_rating_ser) / float(self.batch_size)))

    def __getitem__(self, index):
        x, y = None, None
        if self.hot_encoding:
            x1, x2, y = None, None, None
            if isinstance(index, int):
                x1, x2, y = self.get_batch_data(index)
            elif isinstance(index, slice):
                x1, y, x2 = [], [], []
                for i in range(*index.indices(len(self))):
                    tmp_val = self.get_batch_data(i)
                    x1 += tmp_val[0]
                    x2 += tmp_val[1]
                    y += tmp_val[2]
            return x1, x2, y
        else:
            if isinstance(index, int):
                x = self.get_batch_data(index)
                y = x
            elif isinstance(index, slice):
                x = []
                for i in range(*index.indices(len(self))):
                    x += self.get_batch_data(i)
                y = x

            if not self.weight_sample:
                return x, y
            else:
                weights = self.__get_weight_sampling(x)

            return x, y, weights

    def shuffle(self):
        np.random.shuffle(self.index_list)

    def get_batch_data(self, index, type_of_data: int = 0):
        if self.hot_encoding:
            tmp_list = copy.deepcopy(self.__get_data(index, type_of_data))
            true_value, movie_id = [], []
            for movie, rating in tmp_list:
                rnd_index = np.random.randint(len(movie))
                movie_id.append(movie.pop(rnd_index))
                true_value.append(rating.pop(rnd_index))
            tmp_list = list(map(self._encoding_data, tmp_list))
            return tmp_list, movie_id, true_value
        else:
            tmp_list = self.__get_data(index, type_of_data)
            return tmp_list

    def __get_data(self, index, type_of_data: int = 0):
        data_list = []
        length_check = False
        if type_of_data == self.__split_data_retrieve_dict.get("train"):
            index_type_data = self._train_index_list[index]
            if index_type_data < self.get_train_length():
                length_check = True
        elif type_of_data == self.__split_data_retrieve_dict.get("test"):
            index_type_data = self._test_index_list[index]
            if index_type_data < self.get_test_length():
                length_check = True
        elif type_of_data == self.__split_data_retrieve_dict.get("validation"):
            index_type_data = self._validation_index_list[index]
            if index_type_data < self.get_validation_length():
                length_check = True
        else:
            index_type_data = index

        if length_check:
            for i in range(self.batch_size):
                data_list.append(self.movie_rating_ser.iloc[
                                     index_type_data * self.batch_size + i
                                     ])
        else:
            for i in range(self.batch_size):
                try:
                    data_list.append(self.movie_rating_ser.iloc[
                                         index_type_data * self.batch_size + i
                                         ])
                except:
                    break
        return data_list

    def _encoding_data(self, data):
        assert type(data) == list or type(
            data) == np.numarray, "data is not a list or an array in __encoding_data method"
        ratings = data[1]
        movie_ids = data[0]
        tmp_encoded_list = np.zeros(self.number_of_movies)
        for movie_id, movie_rating in zip(movie_ids, ratings):
            tmp_encoded_list[movie_id - 1] = movie_rating / self.highest_score
        return tmp_encoded_list.tolist()

    def __get_weight_sampling(self, data_list):
        # Gets weights for each row of ratings of the movies of customer id. Weights are used for training.
        # zero is there where there are no ratings for particular movie id, otherwise there is a number
        masked_list = []
        for data in data_list:
            masked_list.append(list(map(lambda x: (x > 0 and 1 or 0), data)))
        return masked_list

    def __get_data_from_training(self):
        try:
            self.movie_rating_ser = pd.read_csv(os.path.join(os.curdir, "saved_data.csv"),
                                                index_col=0)
            self.movie_rating_ser = self.movie_rating_ser.squeeze()
            self.movie_rating_ser = self.movie_rating_ser.apply(ast.literal_eval)
        except:
            list_ids = []
            counter = 0
            # for i in range(20):
            #     file_path = self.dataset.take(1).as_numpy_iterator().next().decode()
            #     with open(file_path, "r") as file:
            #         movie_id = file.readline()
            #         movie_id = int(movie_id[0:-2])       #without last character which is ":"
            #     data_df = pd.read_csv(file_path, sep=",", header=None, skiprows=1)
            #     tmp_list_ids = data_df[0].tolist()
            #     list_ids = list_ids + tmp_list_ids
            #     self.unique_ids = set(list_ids)
            #     self.__put_ratings(tmp_list_ids, data_df[1].tolist(), movie_id=movie_id)
            #     counter += 1
            #     print("Done: {:.2f}".format(counter*100.0/100))
            for file_path in self.dataset:
                file_path_decoded = file_path.numpy().decode()
                with open(file_path_decoded, "r") as file:
                    movie_id = file.readline()
                    movie_id = int(movie_id[0:-2])  # without last character which is ":"
                data_df = pd.read_csv(file_path_decoded, sep=",", header=None, skiprows=1)
                tmp_list_ids = data_df[0].tolist()
                list_ids = list_ids + tmp_list_ids
                self.unique_ids = set(list_ids)
                self.__put_ratings(tmp_list_ids, data_df[1].tolist(), movie_id=movie_id)
                counter += 1
                print("Done: {:.2f}".format(counter * 100.0 / self.number_of_movies))
            self.movie_rating_ser.to_csv("./saved_data.csv")

    def __put_ratings(self, ids: list, ratings: list, movie_id: int):
        for itr, single_id in enumerate(ids):
            tmp_list = [[], []]  # first list in this container is movie id and the second one is rating
            tmp_list[0].append(movie_id)  # adding to first list movie rating
            tmp_list[1].append(ratings[itr])
            try:
                self.movie_rating_ser[single_id][0] = self.movie_rating_ser[single_id][0] + tmp_list[0]
                self.movie_rating_ser[single_id][1] = self.movie_rating_ser[single_id][1] + tmp_list[1]
            except:
                self.movie_rating_ser.loc[single_id] = tmp_list

    def __replace_nan_in_series(self, data_series: list):  # not used any longer, should be used with apply method
        """Replacing NaN values in the list of each index in Series to 0.0"""
        tmp_list = []
        for i, val in enumerate(data_series):
            if (val == np.nan) or (pd.isna(val)):
                tmp_list.append(0.0)
            else:
                tmp_list.append(val)
        return np.array(tmp_list)


class Recommender(MovieDataset):
    def __init__(self, model:tf.keras.Model, data_set:tf.data.Dataset, movie_titles:pd.DataFrame, number_of_movies:int,
                 highest_score:float = 5.0):
        super().__init__(data_set, number_of_movies, highest_score, batch_size=1, weight_sample=False,
                         hot_encoding=True, split_test_validation=False)
        self.model = model
        self.movie_titles_dataframe = movie_titles

    def __raw_output(self, customer_id):
        """This method is for first model. It only takes input of the encoded ratings"""
        return self.model(np.array([self._encoding_data(self.movie_rating_ser.loc[customer_id])]))[0]

    def __raw_output_id(self, idx):
        """This method is for first model. It only takes input of the encoded ratings"""
        return self.model(np.array([self._encoding_data(self.movie_rating_ser.iloc[idx])]))[0]

    def get_movie_title(self, movie_id):
        return self.movie_titles_dataframe['title'][self.movie_titles_dataframe['id'] == movie_id].iloc[0]


    def get_rating(self, user_id, movie_id):
        if self.hot_encoding:
            encoded_ratings = np.array([self._encoding_data(self.movie_rating_ser.loc[user_id])])
            return self.model([np.array([movie_id]), encoded_ratings]).numpy()[0][0]
        else:
            model_output = self.__raw_output(user_id)
            return model_output[movie_id - 1].numpy() * self.highest_score

    def recommend(self, user_id, number_of_movies = 1):
        recommended_movies = pd.DataFrame(columns=["movie_title", "rating"])
        if self.hot_encoding:
            movie_id_list = pd.Series(list(range(1, self.number_of_movies + 1)))
            movie_id_list = movie_id_list.rename(index=lambda x: x + 1)
            movie_id_list = movie_id_list.drop(self.movie_rating_ser.loc[user_id][0], inplace=False)
            ratings = pd.Series()
            for movie_id in movie_id_list:
                ratings.at[movie_id] = self.get_rating(user_id, movie_id)
            ratings = ratings.sort_values(ascending=False)
            if number_of_movies < 6:
                ratings = ratings[:10]
                ratings = ratings.sample(number_of_movies)
            else:
                ratings = ratings[:(2*number_of_movies)]
                ratings = ratings.sample(number_of_movies)

            ratings = ratings.reset_index()
            ratings = ratings.rename({0: 'rating', 'index': 'movie_id'}, axis='columns')
            for id, (movie_id, rating) in enumerate(zip(ratings['movie_id'], ratings['rating'])):
                recommended_movies.loc[id] = {'movie_title': self.get_movie_title(movie_id),
                                                'rating': rating}
        else:
            raw_model_output = pd.Series(np.array(self.__raw_output(user_id)))
            raw_model_output = raw_model_output.rename(index=lambda x: x + 1)
            raw_model_output = raw_model_output.sort_values(ascending=False)
            movie_ids_ratings_series = raw_model_output.iloc[0:number_of_movies]*self.highest_score
            for movie_id, rating in movie_ids_ratings_series.items():
                movie_title = self.movie_titles_dataframe.loc[
                    self.movie_titles_dataframe["id"] == movie_id, ["title"]
                ].iloc[0]["title"]
                new_movie_row = {"movie_title": movie_title, "rating": rating}
                recommended_movies = recommended_movies.append(new_movie_row, ignore_index=True)
        return recommended_movies