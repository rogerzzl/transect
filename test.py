import pandas as pd
import numpy as np


pts = pd.read_excel('topPts.xls',index_col=None, na_values=['NA'], usecols = "C:D")
lis = [1,3]
ptsLis = pts.iloc[lis]
write = pd.ExcelWriter("test.xlsx") 
ptsLis.to_excel(write, sheet_name='Sheet1', index=False)
write.save()
#print(pts.iloc[[:,2],[:,3]])
# In[1]
'''
height,width = ptsLis.shape
x = np.zeros((height,width))
for i in range(0,height):
    for j in range(0,width):
        x[i][j] = ptsLis.iloc[i,j]
print(x)
'''
'''
write = pd.ExcelWriter("test.xlsx")
df1 = pd.DataFrame([1, 2])
df1.to_excel(write, sheet_name='Sheet1', index=False)  

df2 = pd.DataFrame([4, 5]) 
df2.to_excel(write, sheet_name='Sheet2', index=False)
write.save()

print(pts)
'''
# In[2]
def GeneralEquation(first_x,first_y,second_x,second_y):
    A = second_y-first_y
    B = first_x-second_x
    C = second_x*first_y-first_x*second_y
    k = -1 * A / B
    b = -1 * C / B
    print("The line equation is Y={0}X+{1}".format(k,b))
    return k, b

x1, y1 = 5, 0
x2, y2 = 15, 15
A = []
A = GeneralEquation(x1, y1, x2, y2)
x3,y4 = 14, 13.5
b1 = y4 - ((-1/1.5)*x3)
k1 = -1/A[0]
print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

xmax = 18
xmin = 2

ymax = k1 * xmax + b1
ymin = k1 * xmin + b1
# In[3] Try Guilin Yangshuo
####################################
#pt2
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

x3,y3 = 2783342.065, 442111.9402

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

ymax = 442249.3867      

ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1
print("maxX=%.4f , maxy=%.4f, minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))


# In[4] Try Guilin Yangshuo
####################################
#pt3
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

x3,y3 = 2783418.334, 442110.4923

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

ymax = 442249.3867      

ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1
print("maxX=%.4f , maxy=%.4f, minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

# In[5] Try Guilin Yangshuo
####################################
#pt4
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783494.604, 442109.0444


k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442249.3867      
ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1
print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

# In[6] Try Guilin Yangshuo
####################################
#pt5
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783570.874, 442107.5965


k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442249.3867      
ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt5_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt5_bot.csv",index=True,header=True)

# In[7] Try Guilin Yangshuo
####################################
#pt6
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783647.143,442106.1486

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442249.3867      
ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt6_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt6_bot.csv",index=True,header=True)

# In[8] Try Guilin Yangshuo
####################################
#pt7
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783723.413,442104.7007

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442249.3867      
ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt7_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt7_bot.csv",index=True,header=True)

# In[9] Try Guilin Yangshuo
####################################
#pt8
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783799.682,442103.2528

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442249.3867      
ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt8_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt8_bot.csv",index=True,header=True)

# In[10] Try Guilin Yangshuo
####################################
#pt9
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783875.952,442101.8049

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442249.3867      
ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt9_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt9_bot.csv",index=True,header=True)

# In[11] Try Guilin Yangshuo
####################################
#pt10
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783952.222,442100.357

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442249.3867      
ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt10_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt10_bot.csv",index=True,header=True)

# In[12] Try Guilin Yangshuo
####################################
#pt11
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2784028.491, 442098.9091


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2784028.491,442098.9091

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442249.3867      
ymin = 441985.5025

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt11_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt11_bot.csv",index=True,header=True)

# In[13] Try Guilin Yangshuo
####################################
#48-49 pt1
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783265.795, 442113.3881

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt1_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt1_bot.csv",index=True,header=True)

# In[14] Try Guilin Yangshuo
####################################
#48-49 pt2
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783175.197, 442142.7151

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt2_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt2_bot.csv",index=True,header=True)

# In[15] Try Guilin Yangshuo
####################################
#48-49 pt3
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2783084.598, 442172.0421

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt3_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt3_bot.csv",index=True,header=True)

# In[15] Try Guilin Yangshuo
####################################
#48-49 pt4
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2782993.999, 442201.3691

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt4_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt4_bot.csv",index=True,header=True)

# In[16] Try Guilin Yangshuo
####################################
#48-49 pt5
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2782903.401, 442230.6961

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt5_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt5_bot.csv",index=True,header=True)

# In[17] Try Guilin Yangshuo
####################################
#48-49 pt6
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2782812.802, 442260.0231

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt6_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt6_bot.csv",index=True,header=True)

# In[18] Try Guilin Yangshuo
####################################
#48-49 pt7
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2782722.203, 442289.35

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt7_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt7_bot.csv",index=True,header=True)

# In[19] Try Guilin Yangshuo
####################################
#48-49 pt8
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2782631.605, 442318.677

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt8_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt8_bot.csv",index=True,header=True)

# In[20] Try Guilin Yangshuo
####################################
#48-49 pt9
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2782541.006, 442348.004

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt9_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt9_bot.csv",index=True,header=True)

# In[21] Try Guilin Yangshuo
####################################
#48-49 pt10
x1, y1 = 2783265.795, 442113.3881

x2, y2 = 2782359.809, 442406.658


B = GeneralEquation(x1, y1, x2, y2)

###xy in mid line
x3,y3 = 2782450.408, 442377.331

k1 = -1/B[0]
b1 = y3 - (k1 * x3)

print("Second line equa is Y = %.1f X + %.1f" % (k1, b1))

###ymax is top 48's y, ymin is bot 47's y
ymax = 442547.0172      
ymin = 442074.6827

xmax = (ymax-b1)/k1
xmin = (ymin-b1)/k1

print("Top:maxX=%.4f , maxy=%.4f, Bot:minx=%.4f, miny=%.4f" % (xmax,ymax,xmin,ymin))

df = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt3_bot.csv')
df1 = pd.read_csv(r'D:\Projects\Arcpy\transect\guilinyangshuo\47_48\pt4_bot.csv')
df_top = df.replace([1,2,3,4], [xmax,ymax, x3,y3])
df_bot = df1.replace([5,6,7,8], [xmin,ymin, x3,y3])
df_top.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt10_top.csv",index=True,header=True)
df_bot.to_csv(r"D:\Projects\Arcpy\transect\guilinyangshuo\48_49\pt10_bot.csv",index=True,header=True)

# In[22] Distance calculation
####################################

midLine = pd.read_excel(r'D:\Projects\Arcpy\transect\guilinyangshuo\output\47_48MidLine.xls',\
                    index_col=None, na_values=['NA'], usecols = "D:F")
midlist = midLine.values.tolist()

'''
height,width = midLine.shape
x = np.zeros((height,width))
for i in range(0,height):
    for j in range(0,width):
        x[i][j] = midLine.iloc[i,j]

B=[]
for m in range(11):
    for n in range(2):
        B.append[m,n]
print(B)
'''
#增添数据并append

import xlrd
import math
'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt'''
#将excel的两列转化为二个列表

worksheet = xlrd.open_workbook(r'D:\Projects\Arcpy\transect\guilinyangshuo\output\47_48MidLine.xls')
sheet_names= worksheet.sheet_names()
print(sheet_names)

for sheet_name in sheet_names:
    sheet = worksheet.sheet_by_name(sheet_name)
    rows = sheet.nrows # 获取行数
    cols = sheet.ncols # 获取列数，尽管没用到
    all_content1 = []
    for i in range(1,rows) : #excel第一行是列名，从第二列开始取数
        cell1 = sheet.cell_value(i, 0) # 取第一列列数据
        try:
            all_content1.append(cell1)
        except ValueError as e:
            pass
for sheet_name in sheet_names:
    sheet = worksheet.sheet_by_name(sheet_name)
    rows = sheet.nrows # 获取行数
    cols = sheet.ncols # 获取列数，尽管没用到
    all_content2 = []
    for i in range(1,rows) :
        cell2 = sheet.cell_value(i, 1) # 取第二列列数据
        try:
            all_content2.append(cell2)
        except ValueError as e:
            pass    
length1=len(all_content1) #计算数据列表的长度
data=zip(all_content1,all_content2) #使用zip函数将两个列表打包成一个元组


worksheet1 = xlrd.open_workbook(u'C:/Users/Administrator/Desktop/data2.xlsx')
sheet_names1= worksheet1.sheet_names()
#print(sheet_names)

for sheet_name1 in sheet_names1:
    sheet1 = worksheet1.sheet_by_name(sheet_name1)
    rows1 = sheet1.nrows # 获取行数
    cols1 = sheet1.ncols # 获取列数
    all_content3 = []
    for i in range(1,rows1) : #excel第一行是列名，从第二列开始取数
        cell3 = sheet1.cell_value(i, 0) # 取第一列列数据
        try:
            all_content3.append(cell3)
        except ValueError as e:
            pass
for sheet_name2 in sheet_names1:
    sheet2 = worksheet1.sheet_by_name(sheet_name2)
    rows2 = sheet2.nrows # 获取行数
    cols2 = sheet2.ncols # 获取列数，尽管没用到
    all_content4 = []
    for i in range(1,rows2) :
        cell4 = sheet2.cell_value(i, 1) # 取第二列列数据
        try:
            all_content4.append(cell4)
        except ValueError as e:
            pass   
data1=zip(all_content3,all_content4)
#for each in data1:
    #print(each)
length2=len(all_content3)

for i in range(length1):
    x1=all_content1[i]
    y1=all_content2[i]
    j=0
    while j<length2:
        x2=all_content3[j]
        y2=all_content4[j]
        d=math.sqrt(math.pow((x2-x1),2)+math.pow((y2-y1),2)) 
        print('data1表中%d号点与data2表中%d号点之间的距离为:%f'%(i+1,j+1,d))
        j+=1

print(data1)

#End of functions
###################################################################################















