import random
import matplotlib.pyplot as plt
import pandas as pd



cell_df = pd.read_csv(".\\cell_samples.csv")
cell_df.head()

print()









def points(domain):
    x = [random.randint(0,10) for _ in range(domain)]
    y1 = [random.randint(0,10) for _ in range(domain)]
    y2 = [random.randint(0,10) for _ in range(domain)]
    
    return x, y1, y2

domain = 50
x, y1, y2 = points(domain)

f = open('my_points.txt', 'w')

for i in range(domain):
    f.write(f"{x[i]},{y1[i]},0\n")

for i in range(domain):
    f.write(f"{x[i]},{y2[i]},1\n")
    
f.close() 

fig, ax = plt.subplots()


y1_draw, = plt.plot(x, y1, 'o', color='green', label='y1')
y2_draw, = plt.plot(x, y2, 'o', color='red', label='y2')

ax.set_title('random 2 categories points')


ax.legend(handles=[y1_draw, y2_draw])
name = "part1_points" + '.png'

plt.savefig(name)
plt.show()

print()

















Categories = ['2','3','7', 'S', 'W']

flat_data_arr=[]
target_arr=[]

datadir = '.\\persian_LPR'
for i in Categories:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array,(150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target

# df.to_csv('persian_LPR.csv')

x = df.iloc[:,:-1]
# print(x)
y = df.iloc[:,-1]
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify = y)
print('Splitted Successfully')