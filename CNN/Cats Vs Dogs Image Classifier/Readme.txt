First download data-set from this link:- https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data , 
and extract them in current_folder. Description about data is in link,
data-set contains two folders train and test. 
Both folders contains images of dogs and cats respectively. 
In preprocessing step you can do many things for increase dataset size and
it also adds variants into image sets, like rotate images in different angles,
increase and decrease blur level, contrast, convert into gray-scale, convert into RGB and so on,
this step i will leave up to you.

Firstly i just ran though all images into train-set, with image names(dog.jpg, cat.jpg)
classify them and put them into corresponding folders dogs and cats respectively. 
Now we have one main training folders that contains two sub-folders dogs and cats.

We prepare our CNN graph and compile it.

As outcome, result will be 0 or 1. 0(zero) for cat(more catness, less dogness) and 1(one)
for dog(more dogness, less catness).