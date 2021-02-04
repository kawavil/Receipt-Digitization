from django.views import View
from django.shortcuts import render, HttpResponse


class Index(View):

    def get(self, request):
        context = {}
        return render(request, 'index.html', context)

    def post(self, request):
        context = {}
        return render(request, 'index.html', context)


