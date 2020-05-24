fruit_list <- c("Kiwi", "Banana", "Apricot", "Avocado", "Cocos", "Clementine", "Mandarine", "Orange",
                "Limes", "Lemon", "Peach", "Plum", "Raspberry", "Strawberry", "Pineapple", "Pomegranate")

output_n <- length(fruit_list)

#zmiana rozmiaru z 100 x 100 px
img_width <- 20
img_height <- 20
target_size <- c(img_width, img_height)

# RGB = 3 kanały
channels <- 3

# ścieżka
train_image_files_path <- "C:/Users/Admin/Desktop/baza/Trening/"
valid_image_files_path <- "C:/Users/Admin/Desktop/baza/Test"

# ładowanie zdjęć
train_data_gen = image_data_generator( #---------------image_data_generator ładuje zdjęcia TRAINING i skaluje wartości pikseli od 0 do 1
  rescale = 1/255 
)

valid_data_gen <- image_data_generator( #---------------image_data_generator ładuje zdjęcia TEST i skaluje wartości pikseli od 0 do 1
  rescale = 1/255
)  
# ładowanie zdjęć do pamięci i zmiana rozmiaru
train_image_array_gen <- flow_images_from_directory(train_image_files_path, #-----tworzenie tablicy train_image_array_gen i przypisanie do niej zdjęć ze ścieżki do TRAINING 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = fruit_list,
                                                    seed = 42)


valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, #tutaj to samo tylko dla Test
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = fruit_list,
                                                    seed = 42)
cat("Number of images per class:") #---wyświetlenie napisu liczby zdjęć w klasie

table(factor(train_image_array_gen$classes)) #-----factor - zmienna do definiowania grup (w tym przypadku klas), tzw. zmienna wyliczeniowa;
#tworzy tablice z grup wymienionych w train_image_array_gen -> 16 owoców - 16 grup
## 
##   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15 
## 466 490 492 427 490 490 490 479 490 492 492 447 490 492 490 492
cat("\nClass label vs index mapping:\n") #------wyświetla napis
## 
train_image_array_gen$class_indices #---z tablicy train_image_array_gen wyświetla indeksy klasy, czyli które są w kolejności
## $Lemon
## [1] 9 --- Lemon znalazł się pierwszy w tablicy i ma nr 9 w kolejności

fruits_classes_indices <- train_image_array_gen$class_indices #-----przypisanie do fruits_classes_indices indeksów klas
save(fruits_classes_indices, file = "C:/Users/Admin/Desktop/baza/fruits_classes_indices.RData") #--zapisanie indeksów do pliku

# Definiowanie modelu
train_samples <- train_image_array_gen$n #---przypisanie do train_samples z tabeli ze zdjęciami treningowymi

valid_samples <- valid_image_array_gen$n #---przypisanie do valid_samples z tabeli ze zdjęciami testowymi

batch_size <- 32 #-----program bierze po 32 zdjęć treningowych i się uczy, ,np. jak jest 100 zdjęć w treningowym to bierze pierwsze 32 zdj i trenuje, potem następne 32 i trenuje aż do momentu jak wytrenuje wszystkie zdjęcia
epochs <- 10 #-------ilość sekwencji trenowania
# inicjalizacja modelu 
model <- keras_model_sequential()

# dodawanie warstw jest to model splotowej sieci neuronowej - 2 warstwy sieci splotowych, 1 warstwa aktywacyjna i 1 redukująca rozmiar
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>% 
  #wymiar przestrzeni wyjściowej, liczba filtrów, kernel_size - określa wysokość i szerokość okna, padding - wypełnianiem zer w pustych miejscach, input_shape - wielkość na wyjściu,  c() tworzenie wektora
  layer_activation("relu") %>% #------relu to funkcja aktywacji __/   <- jej wykres max(0,x) gdzie x to wejście do neuronu
  
  # 2 warstwa
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%  #--%>% oznacza że argument z lewej strony jest przekazywany do pierwszego argumentu z prawej, Przykład: iris %>% head() is equivalent to head(iris).
  layer_batch_normalization() %>%

  layer_max_pooling_2d(pool_size = c(2,2)) %>% #--tworzy wymiar okna (2,2) które sie przesuwa i sprawdza obszar
  layer_dropout(0.25) %>%
  
  # Spłaszczone i odfiltrowane dane wyjściowe do wektora cech i wprowadzone do gęstej warstwy" cokolwiek to znaczy
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  layer_dense(output_n) %>% 
  layer_activation("softmax") #-- aktywacja funkcją softmax

# compilacja
model %>% compile(
  loss = "categorical_crossentropy", #----ilość strat jaka będzie minimalizowana podczas treningu
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6), #---optimizer to 1 z 2 argumentów potrzebnych do kompilacji modelu, rmsprop -> algorytm: "Utrzymuj ruchomą (dyskontowaną) średnią kwadratu gradientów. Podziel gradient przez pierwiastek z tej średniej", lr - wskaźnik uczenia się, decay - rozkład
  metrics = "accuracy" #---"Metryka to funkcja służąca do oceny wydajności twojego modelu"; accuracy - "Oblicza, jak często prognozy są równe etykietom.", czyli jak dobrze przypisuje rodzaje owoców
)
# początek treningu
hist <- model %>% fit_generator(
  # trening
  train_image_array_gen,
  
  # epoki
  steps_per_epoch = as.integer(train_samples / batch_size), #--steps_per_epoch to liczba całk. próbek/32
  epochs = epochs, 
  # dane testowe
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # wyniki
  verbose = 2,
  callbacks = list(
    # zapisz wynik po epoce 
    callback_model_checkpoint("C:/Users/Admin/Desktop/baza/fruits_checkpoints.h5", save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = "C:/Users/Admin/Desktop/baza/Trening/logs")
  )
)