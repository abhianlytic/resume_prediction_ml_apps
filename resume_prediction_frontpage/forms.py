from django import forms

class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Select a file',
        help_text='max. 10 megabytes'
    )

class File(models.Model):
    upload_file = models.FileField(max_length=254)
    #field_name = models.FileField(upload_to=None, max_length=254, **options)
    upload_date = models.DateTimeField(auto_now_add=True)