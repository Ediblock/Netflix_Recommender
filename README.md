### Netflix recommender


> [!NOTE]
> The code is not very well documented inside but at least the class methods
> are easy to understand. There are few models which I am working on so the code
> contains a lot of commented code which works with other models.
> 
> Currently, the model works with 'model3'
> 
> Model weights are not saved on repo due to large file size.
> In order to work with this model you need to first train it and then use it with 
> seed saved onto your local machine. Data to train it on is provided in the link below

Model created based on data provided by Netflix. This [link](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/data)
contains information about the data and the data itself.

Model has been trained with only 5.6% of all the data. This is done due to large amount
of computational time which it takes to train the model. With only 5.6% of data it is 
approximately 20h to train the model (It will depend on what hardware is used to train model).

Data has been taken randomly, so it is not skewed towards more popular titles.
Moreover, data to test model is also taken randomly, hence it not necessary can be
a good test case for the model. It can contain patterns which model did not see at all
during training phase.

Some movies recommendation from netflix dataset:

Recommendations for user 1847570:


|  no relevant id | movie_title                                       | rating   |
|----------------:|---------------------------------------------------|----------|
|            5215 | Curb Your Enthusiasm: Season 4                    | 4.552688 |
|            1479 | Buffy the Vampire Slayer: Season 2                | 4.555140 |
|            1387 | Six Feet Under: Season 4                          | 4.556361 |   
|            3202 | CSI: Season 2                                     | 4.568326 |           
|            1938 | Lord of the Rings: The Two Towers: Extended Ed... | 4.702519 |

Recommendations for user 2165481:

|  no relevant id | movie_title                            | rating    |
|----------------:|----------------------------------------|-----------|
|            4596 | Seinfeld: Season 4                     | 4.581051  |
|               1 |    Freaks & Geeks: The Complete Series | 4.711543  |
|            1387 | Six Feet Under: Season 4               | 4.620740  |


