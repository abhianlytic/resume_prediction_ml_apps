from django.db import models
from django.conf import settings

# Create your models here.
'''
class File1(models.Model):
    name = models.CharField(max_length=500)
    filepath = models.FileField(upload_to='files/', null=True, verbose_name="")

    def __str__(self):
        return self.name + ": " + str(self.filepath)
'''
class File(models.Model):
    upload_file = models.FileField(max_length=254)
    #field_name = models.FileField(upload_to=None, max_length=254, **options)
    upload_date = models.DateTimeField(auto_now_add=True)


class Contact(models.Model):
    name = models.CharField(max_length=122)
    email = models.CharField(max_length=122)
    phone = models.CharField(max_length=12)
    desc = models.TextField()
    date = models.DateField()

    def __str__(self):
        return self.name