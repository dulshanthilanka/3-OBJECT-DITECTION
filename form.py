from django import forms
from django.contrib.auth.forms import UserChangeForm
from . models import picture

class ImageForm(UserChangeForm):
    image = forms.ImageField()

    class Meta:
        model = picture
        fields = ['image']