from simple_image_download import simple_image_download as simp


main_path =  'C:/Users/alber/Desktop/UniTn/Data Science/Second_Semester/Machine_Learning/Competition/Img_scarp/simple_images'
immagini = ["Hagia Sophia","Guggenheim","Taj Mahal","Dancing House",
"Château de Chenonceau","Niterói Contemporary Art Museum","Pyramids of Giza",
"Acropolis of Athens", "Centre Pompidou", "Gateway Arch", "Musée d’Orsay",
"Gherkin", "Metropolitan Cathedral of Brasília","Mosque of Córdoba","Westminster Abbey",
"Dresden Frauenkirche", "Château Frontenac","Colosseum", "Gardaland","St. Basil’s Cathedral Moscow",
"Dome of the Rock Jerusalem", "Casa Milà", "The White House", "Forbidden City, China",
"Sagrada Familia", "Lincoln Center","Mont-Saint-Michel","Angkor Wat", "Konark Sun Tower","Chrysler Building",
"Sacré-Coeur", "Potala Palace","Louvre Museum", "Sydney Opera House", "Pantheon", 
"Space Needle", "Elizabeth Tower", "Leaning Tower of Pisa", "Eiffel Tower", 
"Heydar Aliyev Center", "Great Mosque of Djenné","Lotus Temple", "La Pedrera",
"Petronas Towers", "Buckingham Palace", "Versailles Palace", "burj al-arab", "fernsehen tower berlin", "winter palace", "arena di verona", "duomo di trento", "chiesa san zeno verona"]

for name in immagini :
    response = simp.simple_image_download
    response().download(name, 10)
    print(response().urls(name, 10))

