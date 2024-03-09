from django.shortcuts import render, redirect
from .form import ImageForm
from .prediction_code import predict

def home(request):
    form = ImageForm()
    
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            result = predict(request.FILES['image'])
            return redirect('view_page', result)
    return render(request, 'home/home.html', {'form': form})

def view_page(request ,result):
    return render(request,'home/view_page.html',{'result':result})