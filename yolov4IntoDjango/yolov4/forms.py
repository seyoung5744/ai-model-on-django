from django import forms
from .models import ImageUploadModel


class UploadImageForm(forms.Form):
  title = forms.CharField(max_length=50)
  #file = forms.FileField()
  image = forms.ImageField()


class ImageUploadForm(forms.ModelForm):
  class Meta:
      model = ImageUploadModel
      fields = ('description', 'document' ) # fields는 모델 클래스의 필드들 중 일부만 폼 클래스에서 사용하고자 할 때 지정하는 옵션