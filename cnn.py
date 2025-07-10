{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5a2c80a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras import datasets,layers,models\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f98ecb15",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(50000, 32, 32, 3)"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()\n",
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "bf35cba6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 32, 32, 3)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56732354",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "fd378bfa",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test = y_test.reshape(-1,)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "0852fec7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[6],\n",
       "       [9],\n",
       "       [9],\n",
       "       [4],\n",
       "       [1]], dtype=uint8)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.shape \n",
    "y_train[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1dfbf891",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "09fb5866",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Here we see there are 50000 training images and 1000 test images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0840875a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x143e11d6c10>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAI4AAACOCAYAAADn/TAIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAWRklEQVR4nO1da4xd1XX+1jnnPmbmzvthT2awsbENDuYNhrahQAip+xJVq7ZBVZVIVVGlVGorKjXKr/ZHJfqn6s8KNTS0zUOkeUAiVEoRJEB4GAi1axvsiZ+D52HP885933N2f8z1WWvtjD2XY/syw+xPQqxz1777nHu8Zq+9npuMMXBw+KjwPu4HcFifcILjkAhOcBwSwQmOQyI4wXFIBCc4DolwWYJDRPuI6AMiGiOir1yph3JY+6Ckfhwi8gEcBfAQgHEA+wE8Yow5fOUez2GtIriM7+4FMGaMOQ4ARPRtAA8DuKjgdPpkBlLLi1zKWuvkNVnf8z0jaOYSaaE3MIKn55B/H2HEdHTJvxs9iRwaii/ac6hLL6N5He08Rz6veHVxu5rhiyCsq3GVkHnG8/XNfX6RtYh/aCBoACBxHVnvSr7WmdCcN8YMwsLlCM4IgDPiehzA3Zf6wkDKw99u6QAADOf0D9mc5ReQJf2iOrP8S3py/Ct9r6bGhRTGtJfSb6MmpsyX+N6lih4XGn7xvqeluyZEYq7A9y5UbQHjcWHXqOLV77gjphd//LLiTQc8z1Q1HdN9hXNq3Im5FM+X61I85HI8R7EY092VohqWKRRiuujrfwtf/CX820ztFFbA5QiOvTAA1h8bABDRowAeBYD+YKWvOKxHXI7gjAO4RlyPAjhrDzLGPAHgCQC4ucs3D2xZ/rzLWhH8FC8JS6Wq4nmGVyMj1vOqtfyWq3ztefqnVerMW6zw54WalvW6mNOaAkJDIF/ii4Je+FAXerJYmFG848+9GNPdZknxjHgWEnPWff2ucrmBmB7LdSrewfkpnj8UK7X1N5sWGq5O1opjVreZLseq2g9gJxFtI6I0gC8AePYy5nNYR0i84hhj6kT05wCeB+ADeNIYc+iKPZnDmsblqCoYY54D8NwVehaHdYTLEpyPijQZjKSW9y9hPVS8sthXF6uaJ63Rqtj+hDWtm0ti72K7p6oha+Ul8bWCvhXklH6gNXko7NSlGvPKdb2BqIhx9Zq2EL2IH3Ixo2+ei3jjkRZznCNtcn/YxSb+4cWC4p2YY+tpu5gjyOg5skaY45f0J6wMF3JwSAQnOA6J0FJVVQ8NZuaXdU05stRAhp1aJZPVXwx4aV5cXOTv1PQcZaGqpOcVAOrC7VQS5mbRWqbr4nupmuVEFA7GilAr5dC6lzBvTajVaZt45LylJhfqPKdH7AAspbX3ebzK6iicLyveUMT/pL0B36DTMsdT4rEyoa2Sr6457rCB4QTHIRGc4DgkQkv3OFV4OEvL0eElX+vtXIqjxtWKVv6FIl8Xl1jWjWUGl8WepGxFzmUYoCr2MRW9BYERe6G0FY6ri8h81Vt5PkBHl2uWOz8QfoKgboVFBkZiOtPP9MLEhH7GOQ4rbNaPj7zH99vazu845VX0wLa2mPSW9DOGkXYhrAS34jgkghMch0RoqaqqwcOEt5wvUjBpxTMzbFaWF3XuSEmoLun/tKO4ZRENrlgqQlqchvhnR9YcJNRO3XKhKqtbZopZWWOBnN/OsxJu8A4/p3jZm26N6Z8TuyTOVXT4vdfwdX7xvOL159itsaWrI6Zz1u80IvRfrmjvM9WcqnK4SnCC45AILVVV5VqEIxPLyUu1qhVJE0lMUaitKk9YMJHwaqYta6YiAneeZRF5Qo15IjHKJ/0KPGH1+JZlJh2qKgBqPS+MUFV2brVQVaavV/FOCCvxzeMnYnpxVieDXd/fH9OdRie9bROqsUOk4Ppl6xmrbGUZo7cG8n1fDG7FcUgEJzgOieAExyERWmuOhxGm5koAgAy0neqL7QRZOjYjaodCYSJHltwbsSexA7y+9NgKppUHDl+Y1lkvpXh1iKh3wM9USen9g5/i73lpPQeFbGaf79Lm+JGJyZg+fux9fl6rtCUbcrL6Tqu0paPEY6siml+vaBM7JfZ8PvQckfVvsxLciuOQCE5wHBKhpaoKIISNW5KVRxtINWN5bKUak5yUpY9kBaVn8VKCF4gKTd9Si/V2fiX1fm0ut4mE5EyWPd9L0GogMHxds5KkSuJn5+uaN32OzW4S3uHOlH4fw2VWR0NGJ3KFhtVTJPRwxS6XFj/bi7QYhC7n2OFqwQmOQyI4wXFIhJbvcfyGietZejTQTUQUT4YPSCSXW80k1N4FVjjCE2Z2KuD9SUev7vZQ7mTzOexqVzwzU2KeyAALrTYkhZD3J1GgTe5KliPW8zW9z8u19cT0tVs50aqtvqjGBSLEsWB1yohEh4ogkpF+/Yyh2GOSHTlvorBq1RWHiJ4komki+j/xWR8RvUBExxr/773UHA6fPDSjqr4OYJ/12VcAvGiM2Qngxca1wwbCqqrKGPMTIrrW+vhhAPc36KcAvAzgb1abyyMg01g9A6tGV3pwIzuH9yIXdhs6eWksXRgKb2hd1GktBTqhbDrP5m026FC8osiLzvZye5GuLcNq3NZtW2N6+JpPK57fx5Ht4quvKV7lPN976gz3rPrw8Ltq3OSmnpheTGlVG0xxYldPntuoSDMdAIxQ3Z7VkiukK6CqLoJNxpgJAGj8fyjhPA7rFFd9cyw7cnU3kefhsD6QVHCmiGjYGDNBRMMApi82UHbkGkl5JtsIvAVWopVMqLIXSuOvvDBaDbkQiSX2F+YQiVeLNRH8szpydey8KaZv+OznFa9/hPv5eTlWY5lurS6k/VIPtWU2U2N1tH3vPYp375YdMX3ojTdj+p/3v6HG/fQkt+Xr7OxWvPu27Y5pc5qTwcKZD9U4qY48S+WHTXSiTaqqngXwxQb9RQDPJJzHYZ2iGXP8WwBeB3A9EY0T0Z8AeBzAQ0R0DMt9jh+/uo/psNbQjFX1yEVYD17hZ3FYR2ip59gDkGnoT2NtlGXylu3J9MTCKB+47tsRXx4XGb0BKvrCW7x9V0wP3HyzGpe5dntMTwd6/3Dw6DjzpnhbV5pbUOPyS/MxPTunk7DmhWf3znvuVLxffuz+mM7dy7/lnXv0Xuh7P/6vmD6/qMuDhzr7Ynqv2DMVF3WHU6/G14HlVa5fxT2OwwaHExyHRGipqiIQgoYHt2qLrIj3kdXhyheeZF/I+rzVuj8lTPwa6a5e3TfsYd7W62L6rXNazcyfZNM3Srcp3qHjx2P69PGxmG63vLKDwqs8MTOreBXRaeve++5TvEKBa6TaOjiv+Fd/+/fUuNcP83EZJ8/8XD/jOHuc023sJqCMVrudFXYL9JJTVQ4tghMch0RwguOQCC1P5KKgkaxumcu+qJ0yVmKUDCXI2vGSVX8eidYdqe3bFW9WuOYPHYxTizA/p8+M6hvgeG29V+8Lwoj3IH5adC61zp1CG6cnpbp1hP2GG2+N6bsf1HucsgiFBEv8226+/ZfUuPsf/PWYfvpb/654psrv7sAY12Z1Brq+a9AXCWtWB642WKearAC34jgkghMch0RoqaoyBEQXPMRW0ydPmNx29kVVdNeqihPh+jZdr8aVyzxuflC3VXznxOmYTgf8s/v6dGR7oJ+vx0O9ZFfrfJ3r4nFehzb9B7ZcG9MP3HGX4j2477dienBkq+JVRR5zIOq2ylYJcFqov5tu1J7vyTE21WdK7B0u9PapcXv28El9gyU9/9zBt7Aa3IrjkAhOcBwSocVWFWAuqCQrz1U7K3XZSC3LOcKzPWyx9O3YocYV6zzJ2Iy2dDbt5gStM6eOxnRonRNqiC2nYlWrqhv3sPd53z7O39+5/Vo1bkQkfPUNaZUpO2ycn9Vea6REXnSVS3G+8fV/VcNe/f73YvqmIX3vsigrnhMly7t371HjPvM5tsyCqSnFe+3QAXFVwkpwK45DIjjBcUgEJzgOidDyPQ41Ith2G5Ka2OTUfP1Ycxkuoz1YEgfFHzqmxrX19MR0V7/eWywW2OQ8JTpfGesNZOfmef453Tj6sb/mKPUfPsKJkdWa3gsZsdcqLumuoJUKd/sM7DJo4XZ47rvfj+k3vvkdNa7tPEfcS0v6BwxvEjVdI7fF9N33PqDGDQ1xLVi6Q7skMt2DfDEzh5XgVhyHRHCC45AIrVVVBnG7JyvGqY7nibp6FG/4Ls65PTjFXavyk9rkri4ItZDWObayGWNVlPkaXwf/+rtFgLJX10R1d3Ny1cQkm9KzeW1Wl0o8v3WkBHpFDVbO8jhLn8TmzXzs0E033qKGFedYfQxt26l4A7tuiOmuQfYW26Vp+SV+d73t+jmi3tV7SLgVxyERnOA4JIITHIdEaPEexwAXkrTqOsG72M8JVHf/wR8pXvaOu2P6pe+wu33puD6rKapzyD3VptuXLC3Mx3RtiTtcZdo71bh20TGrf9OI4vkZ5k3N8HxLJW22yxOje7t0MlhFPOPi1KTi5Tp4T3Xb/Ww+p605xie4DjzVY80vswyEmyAq6wh4JCL/ZybHFW+qZCWmrYBmSoCvIaKXiOgIER0ior9ofO66cm1gNKOq6gAeM8bsBnAPgC8T0afhunJtaDRTOz4B4EITpTwRHQEwgoRduS600KjUtKoave+hmL7rS3+mePtP81LaNcgez1THmBpnRFPpWlWfeqvyguW4ijbbj53gOqVrrtuteF5GJFfV2fS3PcdtQt0V8nr+/37u2Zg+cFB32hrcxB7bX/v8b8b0ddfryHaw6VMxnZ/Xnt2iSPqqCPVU1Q5sVRL82k9eVrzxCa26VsJH2hw3WrrdBuBNuK5cGxpNCw4R5QB8F8BfGmMWVxsvvvcoEb1NRG8XmqgQdFgfaEpwiCiFZaH5hjHmglkz1ejGhUt15TLGPGGMudMYc2eHdVquw/rFqnscIiIAXwNwxBjzj4J1oSvX42iyK1dkgGLj3Mmo3TpgYyu3Hnn+TavL5gLr8R6RdJ0RmYEAQOL0iskPTyuePCI5neHvpbO6gXW7CHek0np+z+cMvarYNNTrVg27SGD84TM/ULz/ePJfYtpYNdsk2s0dPnAwpv/0y3+lxu0Sex6ysiVnRa16qcCKoVbQYZFX/uf5mD7w5k8Vr88O26+AZvw4vwLgjwEcJKL3Gp99FcsC83SjQ9dpAL/fxFwOnxA0Y1W9CqvVsIDryrVB0dq6KhhUG8t6dlB7PF/92Xsx/cOvfVPxbr6do8M7bmE6k9GqpF5iE7xY0Pv3QByF6KXZQ7vn9r1q3NYdHF1ua9PRcV+oKqmeUikdYT83fTamn//RDxQvm2J11Ne/SfFKIkH9+NgHMf3Mf35bjXv4dzmJLJ/XyeQz88KbHnKU/vWXXlDjDrzF6iljHUHd1iHV98peZBerckgEJzgOidBiVQWEjdrfcqS9rafHT8Z0YJ3DkBde33Savbc9IscYAI6d5cbRtbpefjPtvPy297KHtrOnX40riNzkvj7NGxpa2ccZ+NqyOXroZzG9sKA7cvV0clB1bk7zQpHd1iUacB96T1uZu3axR3vzqO7KId/P8Q9Y3R09ckiNy3h8r8FOnXPckbUSzFaAW3EcEsEJjkMiOMFxSISW7nEiAEsNWa0s6eSnaJD3Ndu2XKN4oQhVyGP/2tp0V9BQHDnoW17f7j7e1/Ru5tojY52NVSrwc42OjiqeJ45uLIpG12SFUqZELXZgmeodYo/TntNe6yVx70UR9c7n9V5o7H3uKDYsWqosPwu/nzMnT8Z03Wpl0iPaqGTtTHb7dJUV4FYch0RwguOQCC1VVTUDTIfLsiq7TwFAscJqxmStppBi6SyX2RsaWbG4ikioCtp008Zu0aFrdCubsAO92uQmqQots3Rigj3CRtzcDraG4nnJatroi6Mcu7p7FK8enWO6JNqt5LUX/NQJTmDbMXFK8ZYK/H4+FMczVkVDbECXXBfrOukNVmPwleBWHIdEcILjkAhOcBwSoaV7nJAIixeSlSwT0C+LxKicfdQx7xOKQofnrJDAp7ZxMljXgO6yufMGdtNfv+vGmB7drCPUIpcKmXa9d8mk+TmMbEVnmfQdbWxye6T3OKH4Wx0e0XVbg5t4H3bkALdTK1oJ9ZNTvNc6qtquAYUiR8vPTXP9VdVKNivINSOlQyZIr76euBXHIRGc4DgkQmtLgImAbOOWpG3pVJGX466sXt7zYiWtLrJHdXZ2Ro2DODeqZHmmjx45EtOTp9lMzVne55Qwn+0yYk8kQkaihNmzEiQXznPefmTVXKVT/MqPHT2qePI8i+lz7H2u1LS5nBdtVfa/9oriVaqsyivCWxxYW4OyUK/GOsoy8CzVtQLciuOQCE5wHBKhtUcrEsFreE6zKa0iCmBv6+RpfVxgSeQWnz3DyUmT07qxc2GBE76MtTRLxShVwi/85ZAvSP16ZMNLksWFVqGhJw6qsEuRt2/h8l0iPf/586x6R4aFhfW+/p2RSFJbmNPqWgaBPbEdMNbWAL5Qu7724psm6t/ciuOQCE5wHBLBCY5DIrR2j+N5SDWSxtOB9srKYxFlE2kAmFzkvUtBmKZpq65qcJhboBRKut6oHoljCy9pboq9QKQ92PJadv+KLK9sXdQpRdYch4SnV3qwAWBYeI5Pn+YIeNnqpiUj+Faf8bgBeeOCSV/vW0jUd6WtrqOyFPliaKYjV5aI3iKi/2105Pq7xueuI9cGRjOqqgLgs8aYWwDcCmAfEd0D15FrQ6OZ2nED4IJbN9X4zyBBR66ICJWGSjKWt7UqVFW2T5cHD7dz3Q+Jrli5Lt340Qhv7skTJxSvVGbVlRXeYt+qifJFR2uqaXUXCi+wvFfdOtcqjNh7W7NaYRXKrGoPf6A9x554B4sLnNRlxVCREu/AWOd+qfxnQdq/MxAB23TGapDdRB+jZvvj+I1OFdMAXjDGuI5cGxxNCY4xJjTG3ApgFMBeItqzyldiyI5c1TBc/QsO6wIfyRw3xsxjWSXtQ4KOXGl/9eCZw/pAMx25BgHUjDHzRNQG4HMA/gEJOnIZItQbJrSBjoAHop5706g+Vrl9iBOeakLUC1at0Lxw2ac79D4p18eaVO0RrLOqUyJUEUTazDYiCd2ILti1sk4Er5Y5Ml8uW/VjwruftpPCxZyh6IxaqepWI6K8C56n36PcnkhXQMoKn2RFjblP+g86stqerIRm/DjDAJ4iIh/LK9TTxpgfEdHrcB25NiyasaoOYLlFrf35DFxHrg0LMi1sIUtE5wCcAjAA4PwqwzcS1vL72GqMGbQ/bKngxDcletsYc2fLb7xGsR7fhwtyOiSCExyHRPi4BOeJj+m+axXr7n18LHsch/UPp6ocEqGlgkNE+4joAyIaI6INl4bxSTptsGWqquF5PgrgIQDjAPYDeMQYc7glD7AG0IjpDRtj3iWiTgDvAPgdAF8CMGuMebzxB9VrjFn10LiPE61ccfYCGDPGHDfGVAF8G8s5PRsGxpgJY8y7DToPQJ42+FRj2FNYFqY1jVYKzgiAM+J6vPHZhsR6P22wlYKzUpXXhjTpkp42uJbQSsEZByD70I4COHuRsZ9YXM5pg2sJrRSc/QB2EtE2IkoD+AKWc3o2DJo4bRBoMrfp40aro+O/AeCfAPgAnjTG/H3Lbr4GQESfAfAKgINAXCz/VSzvc54GsAWN3CZjzOyKk6wROM+xQyI4z7FDIjjBcUgEJzgOieAExyERnOA4JIITHIdEcILjkAhOcBwS4f8Bi/i9QI7XFqEAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1080x144 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (15,2))\n",
    "plt.imshow(X_test[6])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "097d4603",
   "metadata": {},
   "outputs": [],
   "source": [
    "classes= [\"airplane\",\"automobile\",\"bird\",\"cat\",\"deer\",\"dog\",\"frog\",\"horse\",\"ship\",\"truck\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "69c3f6d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'truck'"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classes[9]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f0df02e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 59,  62,  63],\n",
       "        [ 43,  46,  45],\n",
       "        [ 50,  48,  43],\n",
       "        ...,\n",
       "        [158, 132, 108],\n",
       "        [152, 125, 102],\n",
       "        [148, 124, 103]],\n",
       "\n",
       "       [[ 16,  20,  20],\n",
       "        [  0,   0,   0],\n",
       "        [ 18,   8,   0],\n",
       "        ...,\n",
       "        [123,  88,  55],\n",
       "        [119,  83,  50],\n",
       "        [122,  87,  57]],\n",
       "\n",
       "       [[ 25,  24,  21],\n",
       "        [ 16,   7,   0],\n",
       "        [ 49,  27,   8],\n",
       "        ...,\n",
       "        [118,  84,  50],\n",
       "        [120,  84,  50],\n",
       "        [109,  73,  42]],\n",
       "\n",
       "       ...,\n",
       "\n",
       "       [[208, 170,  96],\n",
       "        [201, 153,  34],\n",
       "        [198, 161,  26],\n",
       "        ...,\n",
       "        [160, 133,  70],\n",
       "        [ 56,  31,   7],\n",
       "        [ 53,  34,  20]],\n",
       "\n",
       "       [[180, 139,  96],\n",
       "        [173, 123,  42],\n",
       "        [186, 144,  30],\n",
       "        ...,\n",
       "        [184, 148,  94],\n",
       "        [ 97,  62,  34],\n",
       "        [ 83,  53,  34]],\n",
       "\n",
       "       [[177, 144, 116],\n",
       "        [168, 129,  94],\n",
       "        [179, 142,  87],\n",
       "        ...,\n",
       "        [216, 184, 140],\n",
       "        [151, 118,  84],\n",
       "        [123,  92,  72]]], dtype=uint8)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "9532da77",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[[0.61960784, 0.43921569, 0.19215686],\n",
       "         [0.62352941, 0.43529412, 0.18431373],\n",
       "         [0.64705882, 0.45490196, 0.2       ],\n",
       "         ...,\n",
       "         [0.5372549 , 0.37254902, 0.14117647],\n",
       "         [0.49411765, 0.35686275, 0.14117647],\n",
       "         [0.45490196, 0.33333333, 0.12941176]],\n",
       "\n",
       "        [[0.59607843, 0.43921569, 0.2       ],\n",
       "         [0.59215686, 0.43137255, 0.15686275],\n",
       "         [0.62352941, 0.44705882, 0.17647059],\n",
       "         ...,\n",
       "         [0.53333333, 0.37254902, 0.12156863],\n",
       "         [0.49019608, 0.35686275, 0.1254902 ],\n",
       "         [0.46666667, 0.34509804, 0.13333333]],\n",
       "\n",
       "        [[0.59215686, 0.43137255, 0.18431373],\n",
       "         [0.59215686, 0.42745098, 0.12941176],\n",
       "         [0.61960784, 0.43529412, 0.14117647],\n",
       "         ...,\n",
       "         [0.54509804, 0.38431373, 0.13333333],\n",
       "         [0.50980392, 0.37254902, 0.13333333],\n",
       "         [0.47058824, 0.34901961, 0.12941176]],\n",
       "\n",
       "        ...,\n",
       "\n",
       "        [[0.26666667, 0.48627451, 0.69411765],\n",
       "         [0.16470588, 0.39215686, 0.58039216],\n",
       "         [0.12156863, 0.34509804, 0.5372549 ],\n",
       "         ...,\n",
       "         [0.14901961, 0.38039216, 0.57254902],\n",
       "         [0.05098039, 0.25098039, 0.42352941],\n",
       "         [0.15686275, 0.33333333, 0.49803922]],\n",
       "\n",
       "        [[0.23921569, 0.45490196, 0.65882353],\n",
       "         [0.19215686, 0.4       , 0.58039216],\n",
       "         [0.1372549 , 0.33333333, 0.51764706],\n",
       "         ...,\n",
       "         [0.10196078, 0.32156863, 0.50980392],\n",
       "         [0.11372549, 0.32156863, 0.49411765],\n",
       "         [0.07843137, 0.25098039, 0.41960784]],\n",
       "\n",
       "        [[0.21176471, 0.41960784, 0.62745098],\n",
       "         [0.21960784, 0.41176471, 0.58431373],\n",
       "         [0.17647059, 0.34901961, 0.51764706],\n",
       "         ...,\n",
       "         [0.09411765, 0.30196078, 0.48627451],\n",
       "         [0.13333333, 0.32941176, 0.50588235],\n",
       "         [0.08235294, 0.2627451 , 0.43137255]]],\n",
       "\n",
       "\n",
       "       [[[0.92156863, 0.92156863, 0.92156863],\n",
       "         [0.90588235, 0.90588235, 0.90588235],\n",
       "         [0.90980392, 0.90980392, 0.90980392],\n",
       "         ...,\n",
       "         [0.91372549, 0.91372549, 0.91372549],\n",
       "         [0.91372549, 0.91372549, 0.91372549],\n",
       "         [0.90980392, 0.90980392, 0.90980392]],\n",
       "\n",
       "        [[0.93333333, 0.93333333, 0.93333333],\n",
       "         [0.92156863, 0.92156863, 0.92156863],\n",
       "         [0.92156863, 0.92156863, 0.92156863],\n",
       "         ...,\n",
       "         [0.9254902 , 0.9254902 , 0.9254902 ],\n",
       "         [0.9254902 , 0.9254902 , 0.9254902 ],\n",
       "         [0.92156863, 0.92156863, 0.92156863]],\n",
       "\n",
       "        [[0.92941176, 0.92941176, 0.92941176],\n",
       "         [0.91764706, 0.91764706, 0.91764706],\n",
       "         [0.91764706, 0.91764706, 0.91764706],\n",
       "         ...,\n",
       "         [0.92156863, 0.92156863, 0.92156863],\n",
       "         [0.92156863, 0.92156863, 0.92156863],\n",
       "         [0.91764706, 0.91764706, 0.91764706]],\n",
       "\n",
       "        ...,\n",
       "\n",
       "        [[0.34117647, 0.38823529, 0.34901961],\n",
       "         [0.16862745, 0.2       , 0.14509804],\n",
       "         [0.0745098 , 0.09019608, 0.04313725],\n",
       "         ...,\n",
       "         [0.6627451 , 0.72156863, 0.70196078],\n",
       "         [0.71372549, 0.77254902, 0.75686275],\n",
       "         [0.7372549 , 0.79215686, 0.78823529]],\n",
       "\n",
       "        [[0.32156863, 0.37647059, 0.32156863],\n",
       "         [0.18039216, 0.22352941, 0.14117647],\n",
       "         [0.14117647, 0.17254902, 0.08627451],\n",
       "         ...,\n",
       "         [0.68235294, 0.74117647, 0.71764706],\n",
       "         [0.7254902 , 0.78431373, 0.76862745],\n",
       "         [0.73333333, 0.79215686, 0.78431373]],\n",
       "\n",
       "        [[0.33333333, 0.39607843, 0.3254902 ],\n",
       "         [0.24313725, 0.29411765, 0.18823529],\n",
       "         [0.22745098, 0.2627451 , 0.14901961],\n",
       "         ...,\n",
       "         [0.65882353, 0.71764706, 0.69803922],\n",
       "         [0.70588235, 0.76470588, 0.74901961],\n",
       "         [0.72941176, 0.78431373, 0.78039216]]],\n",
       "\n",
       "\n",
       "       [[[0.61960784, 0.74509804, 0.87058824],\n",
       "         [0.61960784, 0.73333333, 0.85490196],\n",
       "         [0.54509804, 0.65098039, 0.76078431],\n",
       "         ...,\n",
       "         [0.89411765, 0.90588235, 0.91764706],\n",
       "         [0.92941176, 0.9372549 , 0.95294118],\n",
       "         [0.93333333, 0.94509804, 0.96470588]],\n",
       "\n",
       "        [[0.66666667, 0.78431373, 0.89803922],\n",
       "         [0.6745098 , 0.78039216, 0.88627451],\n",
       "         [0.59215686, 0.69019608, 0.78823529],\n",
       "         ...,\n",
       "         [0.90980392, 0.90980392, 0.9254902 ],\n",
       "         [0.96470588, 0.96470588, 0.98039216],\n",
       "         [0.96470588, 0.96862745, 0.98431373]],\n",
       "\n",
       "        [[0.68235294, 0.78823529, 0.88235294],\n",
       "         [0.69019608, 0.78431373, 0.87058824],\n",
       "         [0.61568627, 0.70196078, 0.78039216],\n",
       "         ...,\n",
       "         [0.90196078, 0.89803922, 0.90980392],\n",
       "         [0.98039216, 0.97647059, 0.98431373],\n",
       "         [0.96078431, 0.95686275, 0.96862745]],\n",
       "\n",
       "        ...,\n",
       "\n",
       "        [[0.12156863, 0.15686275, 0.17647059],\n",
       "         [0.11764706, 0.15294118, 0.17254902],\n",
       "         [0.10196078, 0.1372549 , 0.15686275],\n",
       "         ...,\n",
       "         [0.14509804, 0.15686275, 0.18039216],\n",
       "         [0.03529412, 0.05098039, 0.05490196],\n",
       "         [0.01568627, 0.02745098, 0.01960784]],\n",
       "\n",
       "        [[0.09019608, 0.13333333, 0.15294118],\n",
       "         [0.10588235, 0.14901961, 0.16862745],\n",
       "         [0.09803922, 0.14117647, 0.16078431],\n",
       "         ...,\n",
       "         [0.0745098 , 0.07843137, 0.09411765],\n",
       "         [0.01568627, 0.02352941, 0.01176471],\n",
       "         [0.01960784, 0.02745098, 0.01176471]],\n",
       "\n",
       "        [[0.10980392, 0.16078431, 0.18431373],\n",
       "         [0.11764706, 0.16862745, 0.19607843],\n",
       "         [0.1254902 , 0.17647059, 0.20392157],\n",
       "         ...,\n",
       "         [0.01960784, 0.02352941, 0.03137255],\n",
       "         [0.01568627, 0.01960784, 0.01176471],\n",
       "         [0.02745098, 0.03137255, 0.02745098]]],\n",
       "\n",
       "\n",
       "       ...,\n",
       "\n",
       "\n",
       "       [[[0.07843137, 0.05882353, 0.04705882],\n",
       "         [0.0745098 , 0.05490196, 0.04313725],\n",
       "         [0.05882353, 0.05490196, 0.04313725],\n",
       "         ...,\n",
       "         [0.03921569, 0.03529412, 0.02745098],\n",
       "         [0.04705882, 0.04313725, 0.03529412],\n",
       "         [0.05098039, 0.04705882, 0.03921569]],\n",
       "\n",
       "        [[0.08235294, 0.0627451 , 0.05098039],\n",
       "         [0.07843137, 0.0627451 , 0.05098039],\n",
       "         [0.07058824, 0.06666667, 0.04705882],\n",
       "         ...,\n",
       "         [0.03921569, 0.03529412, 0.02745098],\n",
       "         [0.03921569, 0.03529412, 0.02745098],\n",
       "         [0.04705882, 0.04313725, 0.03529412]],\n",
       "\n",
       "        [[0.08235294, 0.0627451 , 0.05098039],\n",
       "         [0.08235294, 0.06666667, 0.04705882],\n",
       "         [0.07843137, 0.07058824, 0.04313725],\n",
       "         ...,\n",
       "         [0.04705882, 0.04313725, 0.03529412],\n",
       "         [0.04705882, 0.04313725, 0.03529412],\n",
       "         [0.05098039, 0.04705882, 0.03921569]],\n",
       "\n",
       "        ...,\n",
       "\n",
       "        [[0.12941176, 0.09803922, 0.05098039],\n",
       "         [0.13333333, 0.10196078, 0.05882353],\n",
       "         [0.13333333, 0.10196078, 0.05882353],\n",
       "         ...,\n",
       "         [0.10980392, 0.09803922, 0.20392157],\n",
       "         [0.11372549, 0.09803922, 0.22745098],\n",
       "         [0.09019608, 0.07843137, 0.16470588]],\n",
       "\n",
       "        [[0.12941176, 0.09803922, 0.05490196],\n",
       "         [0.13333333, 0.10196078, 0.05882353],\n",
       "         [0.13333333, 0.10196078, 0.05882353],\n",
       "         ...,\n",
       "         [0.10588235, 0.09411765, 0.20392157],\n",
       "         [0.10588235, 0.09411765, 0.21960784],\n",
       "         [0.09803922, 0.08627451, 0.18431373]],\n",
       "\n",
       "        [[0.12156863, 0.09019608, 0.04705882],\n",
       "         [0.1254902 , 0.09411765, 0.05098039],\n",
       "         [0.12941176, 0.09803922, 0.05490196],\n",
       "         ...,\n",
       "         [0.09411765, 0.09019608, 0.19607843],\n",
       "         [0.10196078, 0.09019608, 0.20784314],\n",
       "         [0.09803922, 0.07843137, 0.18431373]]],\n",
       "\n",
       "\n",
       "       [[[0.09803922, 0.15686275, 0.04705882],\n",
       "         [0.05882353, 0.14117647, 0.01176471],\n",
       "         [0.09019608, 0.16078431, 0.07058824],\n",
       "         ...,\n",
       "         [0.23921569, 0.32156863, 0.30588235],\n",
       "         [0.36078431, 0.44313725, 0.43921569],\n",
       "         [0.29411765, 0.34901961, 0.36078431]],\n",
       "\n",
       "        [[0.04705882, 0.09803922, 0.02352941],\n",
       "         [0.07843137, 0.14509804, 0.02745098],\n",
       "         [0.09411765, 0.14117647, 0.05882353],\n",
       "         ...,\n",
       "         [0.45098039, 0.5254902 , 0.54117647],\n",
       "         [0.58431373, 0.65882353, 0.69411765],\n",
       "         [0.40784314, 0.45882353, 0.51372549]],\n",
       "\n",
       "        [[0.04705882, 0.09803922, 0.04313725],\n",
       "         [0.05882353, 0.11372549, 0.02352941],\n",
       "         [0.13333333, 0.15686275, 0.09411765],\n",
       "         ...,\n",
       "         [0.60392157, 0.6745098 , 0.71372549],\n",
       "         [0.61568627, 0.68627451, 0.75294118],\n",
       "         [0.45490196, 0.50588235, 0.59215686]],\n",
       "\n",
       "        ...,\n",
       "\n",
       "        [[0.39215686, 0.50588235, 0.31764706],\n",
       "         [0.40392157, 0.51764706, 0.32941176],\n",
       "         [0.40784314, 0.5254902 , 0.3372549 ],\n",
       "         ...,\n",
       "         [0.38039216, 0.50196078, 0.32941176],\n",
       "         [0.38431373, 0.49411765, 0.32941176],\n",
       "         [0.35686275, 0.4745098 , 0.30980392]],\n",
       "\n",
       "        [[0.40392157, 0.51764706, 0.3254902 ],\n",
       "         [0.40784314, 0.51372549, 0.3254902 ],\n",
       "         [0.41960784, 0.52941176, 0.34117647],\n",
       "         ...,\n",
       "         [0.39607843, 0.51764706, 0.34117647],\n",
       "         [0.38823529, 0.49803922, 0.32941176],\n",
       "         [0.36078431, 0.4745098 , 0.30980392]],\n",
       "\n",
       "        [[0.37254902, 0.49411765, 0.30588235],\n",
       "         [0.37254902, 0.48235294, 0.29803922],\n",
       "         [0.39607843, 0.50196078, 0.31764706],\n",
       "         ...,\n",
       "         [0.36470588, 0.48627451, 0.31372549],\n",
       "         [0.37254902, 0.48235294, 0.31764706],\n",
       "         [0.36078431, 0.47058824, 0.31372549]]],\n",
       "\n",
       "\n",
       "       [[[0.28627451, 0.30588235, 0.29411765],\n",
       "         [0.38431373, 0.40392157, 0.44313725],\n",
       "         [0.38823529, 0.41568627, 0.44705882],\n",
       "         ...,\n",
       "         [0.52941176, 0.58823529, 0.59607843],\n",
       "         [0.52941176, 0.58431373, 0.60392157],\n",
       "         [0.79607843, 0.84313725, 0.8745098 ]],\n",
       "\n",
       "        [[0.27058824, 0.28627451, 0.2745098 ],\n",
       "         [0.32941176, 0.34901961, 0.38039216],\n",
       "         [0.26666667, 0.29411765, 0.31764706],\n",
       "         ...,\n",
       "         [0.33333333, 0.37254902, 0.34901961],\n",
       "         [0.27843137, 0.32156863, 0.31372549],\n",
       "         [0.47058824, 0.52156863, 0.52941176]],\n",
       "\n",
       "        [[0.27058824, 0.28627451, 0.2745098 ],\n",
       "         [0.35294118, 0.37254902, 0.39215686],\n",
       "         [0.24313725, 0.27843137, 0.29019608],\n",
       "         ...,\n",
       "         [0.29019608, 0.31764706, 0.2745098 ],\n",
       "         [0.20784314, 0.24313725, 0.21176471],\n",
       "         [0.24313725, 0.29019608, 0.27058824]],\n",
       "\n",
       "        ...,\n",
       "\n",
       "        [[0.48235294, 0.50196078, 0.37647059],\n",
       "         [0.51764706, 0.51764706, 0.4       ],\n",
       "         [0.50588235, 0.50196078, 0.39215686],\n",
       "         ...,\n",
       "         [0.42352941, 0.41960784, 0.34509804],\n",
       "         [0.24313725, 0.23529412, 0.21568627],\n",
       "         [0.10588235, 0.10588235, 0.10980392]],\n",
       "\n",
       "        [[0.45098039, 0.4745098 , 0.35686275],\n",
       "         [0.48235294, 0.48627451, 0.37254902],\n",
       "         [0.50588235, 0.49411765, 0.38823529],\n",
       "         ...,\n",
       "         [0.45098039, 0.45490196, 0.36862745],\n",
       "         [0.25882353, 0.25490196, 0.23137255],\n",
       "         [0.10588235, 0.10588235, 0.10588235]],\n",
       "\n",
       "        [[0.45490196, 0.47058824, 0.35294118],\n",
       "         [0.4745098 , 0.47843137, 0.36862745],\n",
       "         [0.50588235, 0.50196078, 0.39607843],\n",
       "         ...,\n",
       "         [0.45490196, 0.45098039, 0.36862745],\n",
       "         [0.26666667, 0.25490196, 0.22745098],\n",
       "         [0.10588235, 0.10196078, 0.10196078]]]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Normalizing the training data\n",
    "X_train = X_train / 255.0\n",
    "X_test = X_test / 255.0\n",
    "\n",
    "X_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55ab5345",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n"
     ]
    }
   ],
   "source": [
    "#Build simple artificial neural network for image classification\n",
    "ann = models.Sequential([\n",
    "        layers.Flatten(input_shape=(32,32,3)),\n",
    "        layers.Dense(3000, activation='relu'),\n",
    "        layers.Dense(1000, activation='relu'),\n",
    "        layers.Dense(10, activation='softmax')    \n",
    "    ])\n",
    "\n",
    "ann.compile(optimizer='SGD',  \n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "ann.fit(X_train, y_train, epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc916880",
   "metadata": {},
   "outputs": [],
   "source": [
    "#You can see that at the end of 5 epochs, accuracy is at around 49%\n",
    "\n",
    "from sklearn.metrics import confusion_matrix , classification_report\n",
    "import numpy as np\n",
    "y_pred = ann.predict(X_test)\n",
    "y_pred_classes = [np.argmax(element) for element in y_pred]\n",
    "\n",
    "print(\"Classification Report: \\n\", classification_report(y_test, y_pred_classes))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b44aeea5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Now let us build a convolutional neural network to train our images\n",
    "cnn = models.Sequential([\n",
    "    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    \n",
    "    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    \n",
    "    layers.Flatten(),\n",
    "    layers.Dense(64, activation='relu'),\n",
    "    layers.Dense(10, activation='softmax')\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7065d94c",
   "metadata": {},
   "outputs": [],
   "source": [
    "cnn.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "cnn.fit(X_train, y_train, epochs=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b207cfb",
   "metadata": {},
   "outputs": [],
   "source": [
    "cnn.evaluate(X_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d50380f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = cnn.predict(X_test)\n",
    "y_pred[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c26ed992",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_classes = [np.argmax(element) for element in y_pred]\n",
    "y_classes[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c7585d6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "356951d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "classes[y_classes[3]]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d01bd30",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
