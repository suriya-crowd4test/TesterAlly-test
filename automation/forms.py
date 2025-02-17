from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Select an image',
        help_text='Upload an image to analyze UI elements'
    )