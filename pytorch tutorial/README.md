followed official documentation: 
https://docs.pytorch.org/tutorials/beginner/basics/intro.html

# matplotlib plt.show() does not work at first
this fixed it: You have to install tk first.
``` 
import matplotlib
matplotlib.use('TkAgg') # or 'Qt5Agg' if PyQt5 is installed
import matplotlib.pyplot as plt
```