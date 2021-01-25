import os
os.chdir("./ann_basic/")

data = from ann_basic.data.data import

# load data


# pre pros data

item_data = data[]

dog_list = ["tasha", "coopy", "firualais", "quilla", "melo", "tarzan", "boby"]
cat_list = ["maya", "perlita", "tutu", "spyke", "teodoro", "pancito", "chanchirri"]

dog_list_last_name = []

for dog in dog_list:
    dog_with_last_name = dog + " " + "diana"
    dog_list_last_name.append(dog_with_last_name)

cat_list_last_name = []

for cat in cat_list:
    cat_with_last_name = cat + " " + "chunguis"
    cat_list_last_name.append(cat_with_last_name)

name_list = ["mariana", "pedro", "karina"]

last_name_list = ["andrade", "gonzales", "diaz"]

full_name_list = []

for name in name_list:
    name_with_last_name = name + " " + str(last_name_list) ## EYE
    full_name_list.append(name_with_last_name)

name_list[1]

for i in range(3):
    name = name_list[i]
    print(name)
    last_name = last_name_list[i]
    print(last_name)
    full_name = name + " " + last_name
    print(full_name)

full_name_list = []

for i in range(3):
    name = name_list[i]
    last_name = last_name_list[i]
    full_name = name + " " + last_name
    print(full_name)
    full_name_list.append(full_name)

list_dog_added_cat = []

for i in range(6):
    dog = dog_list[i]
    cat = cat_list[i]
    dog_added_cat = dog + " " + cat
    list_dog_added_cat.append(dog_added_cat)

## list of list
image = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
]

## list of arrays

image2 = [
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
        np.array([0, 0, 0])
]

## array of array (neuron understand array the array did not list)

image3 = np.array([
        np.array([0, 0, 0]).flatten(),
        np.array([0, 0, 0]).flatten(),
        np.array([0, 0, 0]).flatten()
])

## NAMES

name1_list = ["Diana", "Juan", "Mel", "Vale", "Kimi"]