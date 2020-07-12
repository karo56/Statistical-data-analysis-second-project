raport
================
Karol Mućk
17 05 2020

Celem projektu jest przeanalizowanie zbioru zawierającego ekspresję
genów w liniach komórkowych oraz działanie leku na nowotworowe linie
komórkowe oraz stworzenie i wytrenowanie modelu który przewidwywał by
działanie leku na innych danych.

Wczytanie danych:

``` r
load("cancer.RData")

x <- as.matrix(data.train[,-1])
y <- (data.train$Y)

dim(x)
```

    ## [1]   643 17737

<b> Zadanie 1 </b>

Zbiór danych zawiera 17737 zmiennych, z czego wszystkie są zmiennymi
ilościowymi. Do wyboru zmiennych o największej zmienności skorzystamy z
współczynnika zmienności wyliczonego ze wzoru
\(V = \frac{s}{\overline{x}}\), \(\overline{x} \ne 0\). Gdzie \(s\) jest
odchyleniem standardowym z próby, natomiast \(\overline{x}\) jest
średnią artymetyczną z próby.

``` r
n = ncol(x)
coeff_Var = sd(x[,1]) / mean(x[,1])

for(i in 2:n)
{
  var =  sd(x[,i]) / mean(x[,i])
  coeff_Var = c(coeff_Var, var)
}
```

Teraz wybierzemy te zmienne które mają największy współczynnik.

``` r
index <- which(coeff_Var >= sort(coeff_Var, decreasing=T)[500], arr.ind=TRUE)
buff = x[,index]
```

Oraz policzmy macierz korelacji dla tych zmiennych.

``` r
res <- cor(buff)
res <-round(res, 2)
```

Wykres wygląda następująco:

``` r
all_res <- c(res)
zeros = replicate(500, 0)

df <- data.frame(zeros,all_res)

ggplot(df, aes(x=zeros, y=all_res)) + 
  geom_violin(trim=FALSE,fill = "#00AFBB", color= "#00AFBB") +theme_minimal()
```

![](Raport_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Warto tu dodać, że macierz zawiera wszystkie wartości dwukrotnie, oraz
na diagonali mamy same 1, ale nie przeszkadza to na wykresie, dlatego
nie pozbyłem się ich.

<b> Zadanie 2 </b>

Rregrsja Elastic Net łączy w sobie cechy regresji grzbietowej oraz
lasso. Metoda polega na minimalizacji następującego wzoru:
\[\hat{\beta} = arg\min_{\beta}||y -X \beta||^{2}+\lambda(\alpha ||\beta||^{2} +(1-\alpha) ||\beta||_{1})\]
Dla \(\alpha = 0\) metoda jest równoważna metodzie grzbietowej,
natomiast dla \(\alpha = 1\) metodzie lasso.

<b> Zadanie 3</b>

Do wybrania zbioru najlepszych predyktorów zastosujemy metodę analizy
grafów (metoda Bartosiewicz) \[“Procedura”stopniowego" konstruowania
liniowych modeli ekonometrycznych o wielu zmiennych objaśniających",
Stanisława Bartosiewicz 2007\]. Metoda działa w następujących etatpach:

1.  wyznaczamy wartość krytyczną
    \(r^{*} = \sqrt{\frac{t_{\alpha,n-2}^{2}}{t_{\alpha,n-2}^{2}+ n-2}}\),
    gdzie \(t_{\alpha,n-2}\) jest wartością testu t Studenta dla poziomu
    istotność \(\alpha\) oraz n-2 stopni swobody.
2.  Eliminuje się ze zbioru zmiennych te zmienne dla których korelacja
    jest zmienną objaśnianą jest mniejsza od krytycznej \(r^{*}\)
3.  Z pozostałych zmiennych wybieramy taką zmienną \((X_{h})\), dla
    której korelacja ze zmienną objaśnianą jest największa.
4.  Eliminuje się ze zbioru potencjalnych zmiennych te wszystkie, dla
    ktorych korelacja ze zmienną \(X_{h}\), jest większa od krtycznej.

Sekwencję możemy powtarzać, aż zostanie optymalna dla nas liczba
zmiennych.

<b> Zadanie 4</b>

Zobaczmy wymiar zbioru danych.

``` r
dim(data.train)
```

    ## [1]   643 17738

Zbiór danych ma bardzo dużo zmiennych, dlatego obetniemy zbiór do
zmiennych które okazały się najlepsze stosując metodę z porzedniego
zadania.

``` r
n = ncol(x)
m = nrow(x)
t_alpha = qt(0.995, df=m-2) 
r_star = sqrt(t_alpha^2 / (t_alpha^2 + m - 2))


R0 = replicate(n, 0)

for (i in 1:n )
{
  R0[i] = cor(x[,i],data.train$Y, method = "pearson")
}

index = c()

for (i in 1:n )
{
  if (abs(R0[i]) <= r_star)
  {
    index = c(index,i)
    
  }
}

R0 = R0[-length(R0)]
absmax <- function(x) { x[which.max( abs(x) )]}
rh = which(R0==absmax(R0))


Rhi = replicate(n, 0)

for (i in 1:n )
{
  Rhi[i] = cor(x[,i],x[,rh], method = "pearson")
}


for (i in 1:n )
{
  if (abs(Rhi[i]) > r_star & !is.element(i, index))
  {
    index = c(index,i)
    
  }
}

data.train = subset(data.train, select = -index)

dim(data.train)
```

    ## [1] 643 291

Dzięki tej metodzie zostało nam tylko 291 zmiennych.

Aby znaleźć najlepsze parametry dla modelu lasu losowego oraz regresji
elastic net zastosujemy metodę wyszykiwania po siatce (ang. Grid
Search), polegajacej na testowaniu wiele kombinacji parametrów
tuningowych. Do znalezienia odpowiednich wartości zastosujemy funckje
glmnet. Z uwagii, że predyckje w zadniu 5 będą ocenianie na pomocą błędu
średniokwadratowego (RMSE), to będzie dla nas najbardziej istotna
informacja.

Do eksperymentów ustawimy stałe zairno, tak, aby losowość dla każdej
próby była “taka sama”.

``` r
set.seed(100)
```

Teraz dostroimy model regresji elastic net.

``` r
cv_5 = trainControl(method = "cv", number = 5)
model <- train(Y ~., data = data.train, method = "glmnet",trControl = trainControl("cv", number = 10),tuneLength = 10)
```

Zobaczmy jak wyglądają wyniki:

``` r
model
```

    ## glmnet 
    ## 
    ## 643 samples
    ## 290 predictors
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 579, 579, 578, 579, 579, 578, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   alpha  lambda        RMSE        Rsquared   MAE       
    ##   0.1    1.322499e-05  0.09220100  0.1033979  0.06601947
    ##   0.1    3.055145e-05  0.09220082  0.1033992  0.06601881
    ##   0.1    7.057781e-05  0.09186735  0.1037861  0.06567968
    ##   0.1    1.630439e-04  0.09105060  0.1046427  0.06487141
    ##   0.1    3.766526e-04  0.08935281  0.1064508  0.06315436
    ##   0.1    8.701163e-04  0.08620772  0.1098790  0.05990904
    ##   0.1    2.010082e-03  0.08126485  0.1156003  0.05448044
    ##   0.1    4.643549e-03  0.07532837  0.1251175  0.04769137
    ##   0.1    1.072720e-02  0.07004841  0.1371134  0.04013282
    ##   0.1    2.478123e-02  0.06734896  0.1417660  0.03401297
    ##   0.2    1.322499e-05  0.09221751  0.1033719  0.06603913
    ##   0.2    3.055145e-05  0.09212449  0.1035113  0.06594160
    ##   0.2    7.057781e-05  0.09153024  0.1041292  0.06536087
    ##   0.2    1.630439e-04  0.09036852  0.1053969  0.06420955
    ##   0.2    3.766526e-04  0.08801501  0.1081311  0.06185367
    ##   0.2    8.701163e-04  0.08381131  0.1129941  0.05738591
    ##   0.2    2.010082e-03  0.07802932  0.1218490  0.05094919
    ##   0.2    4.643549e-03  0.07201821  0.1347853  0.04319515
    ##   0.2    1.072720e-02  0.06832209  0.1431370  0.03603277
    ##   0.2    2.478123e-02  0.06710393  0.1566325  0.03203431
    ##   0.3    1.322499e-05  0.09222889  0.1033859  0.06605208
    ##   0.3    3.055145e-05  0.09195573  0.1036992  0.06578289
    ##   0.3    7.057781e-05  0.09122711  0.1044936  0.06507608
    ##   0.3    1.630439e-04  0.08973517  0.1061972  0.06357958
    ##   0.3    3.766526e-04  0.08675084  0.1097714  0.06059871
    ##   0.3    8.701163e-04  0.08185601  0.1157540  0.05520645
    ##   0.3    2.010082e-03  0.07567107  0.1272661  0.04819315
    ##   0.3    4.643549e-03  0.07026199  0.1408832  0.04020551
    ##   0.3    1.072720e-02  0.06783734  0.1478096  0.03382172
    ##   0.3    2.478123e-02  0.06700551  0.1746920  0.03153293
    ##   0.4    1.322499e-05  0.09223157  0.1034314  0.06605607
    ##   0.4    3.055145e-05  0.09180458  0.1039025  0.06563848
    ##   0.4    7.057781e-05  0.09092236  0.1049058  0.06477997
    ##   0.4    1.630439e-04  0.08910433  0.1070369  0.06296298
    ##   0.4    3.766526e-04  0.08558817  0.1113100  0.05939160
    ##   0.4    8.701163e-04  0.08019362  0.1188731  0.05342712
    ##   0.4    2.010082e-03  0.07393026  0.1317154  0.04586583
    ##   0.4    4.643549e-03  0.06936782  0.1432036  0.03821965
    ##   0.4    1.072720e-02  0.06753318  0.1539338  0.03271359
    ##   0.4    2.478123e-02  0.06729072  0.1793285  0.03153684
    ##   0.5    1.322499e-05  0.09219424  0.1034840  0.06601514
    ##   0.5    3.055145e-05  0.09167721  0.1040252  0.06551687
    ##   0.5    7.057781e-05  0.09064843  0.1052512  0.06450534
    ##   0.5    1.630439e-04  0.08849234  0.1078269  0.06237550
    ##   0.5    3.766526e-04  0.08452777  0.1125978  0.05824473
    ##   0.5    8.701163e-04  0.07877138  0.1218453  0.05185252
    ##   0.5    2.010082e-03  0.07256953  0.1354993  0.04391265
    ##   0.5    4.643549e-03  0.06890561  0.1436625  0.03670560
    ##   0.5    1.072720e-02  0.06735000  0.1642221  0.03216994
    ##   0.5    2.478123e-02  0.06753496  0.1878872  0.03177591
    ##   0.6    1.322499e-05  0.09212680  0.1035465  0.06594821
    ##   0.6    3.055145e-05  0.09153400  0.1042026  0.06537702
    ##   0.6    7.057781e-05  0.09035255  0.1055844  0.06421066
    ##   0.6    1.630439e-04  0.08792379  0.1085809  0.06181341
    ##   0.6    3.766526e-04  0.08357612  0.1138469  0.05715634
    ##   0.6    8.701163e-04  0.07753869  0.1243532  0.05043813
    ##   0.6    2.010082e-03  0.07156671  0.1386625  0.04229554
    ##   0.6    4.643549e-03  0.06857675  0.1458724  0.03547017
    ##   0.6    1.072720e-02  0.06728146  0.1739845  0.03178180
    ##   0.6    2.478123e-02  0.06763864  0.2004176  0.03193105
    ##   0.7    1.322499e-05  0.09205299  0.1036127  0.06587804
    ##   0.7    3.055145e-05  0.09139741  0.1043498  0.06524594
    ##   0.7    7.057781e-05  0.09007579  0.1059206  0.06393157
    ##   0.7    1.630439e-04  0.08734878  0.1093153  0.06124907
    ##   0.7    3.766526e-04  0.08267736  0.1152111  0.05616053
    ##   0.7    8.701163e-04  0.07647328  0.1267841  0.04920886
    ##   0.7    2.010082e-03  0.07086467  0.1407837  0.04107709
    ##   0.7    4.643549e-03  0.06831061  0.1495905  0.03447628
    ##   0.7    1.072720e-02  0.06730614  0.1786405  0.03161408
    ##   0.7    2.478123e-02  0.06783396  0.2024875  0.03208316
    ##   0.8    1.322499e-05  0.09197980  0.1037111  0.06580730
    ##   0.8    3.055145e-05  0.09126972  0.1045348  0.06511859
    ##   0.8    7.057781e-05  0.08978411  0.1062975  0.06364409
    ##   0.8    1.630439e-04  0.08680362  0.1100421  0.06070128
    ##   0.8    3.766526e-04  0.08186844  0.1165777  0.05529587
    ##   0.8    8.701163e-04  0.07558152  0.1287102  0.04807950
    ##   0.8    2.010082e-03  0.07035640  0.1419797  0.04011713
    ##   0.8    4.643549e-03  0.06809972  0.1522829  0.03376754
    ##   0.8    1.072720e-02  0.06739646  0.1791812  0.03151791
    ##   0.8    2.478123e-02  0.06808814  0.2024875  0.03221989
    ##   0.9    1.322499e-05  0.09191385  0.1037824  0.06574181
    ##   0.9    3.055145e-05  0.09114169  0.1047072  0.06498991
    ##   0.9    7.057781e-05  0.08952149  0.1066206  0.06338987
    ##   0.9    1.630439e-04  0.08627105  0.1107131  0.06015918
    ##   0.9    3.766526e-04  0.08109999  0.1179788  0.05447504
    ##   0.9    8.701163e-04  0.07478991  0.1305206  0.04702017
    ##   0.9    2.010082e-03  0.06995758  0.1427968  0.03919565
    ##   0.9    4.643549e-03  0.06785735  0.1553402  0.03319675
    ##   0.9    1.072720e-02  0.06749933  0.1796626  0.03151275
    ##   0.9    2.478123e-02  0.06842617  0.1867297  0.03238389
    ##   1.0    1.322499e-05  0.09185680  0.1038518  0.06568811
    ##   1.0    3.055145e-05  0.09101535  0.1048516  0.06487029
    ##   1.0    7.057781e-05  0.08923790  0.1069812  0.06311493
    ##   1.0    1.630439e-04  0.08577287  0.1113790  0.05963417
    ##   1.0    3.766526e-04  0.08038099  0.1194917  0.05369716
    ##   1.0    8.701163e-04  0.07405748  0.1323508  0.04600230
    ##   1.0    2.010082e-03  0.06962687  0.1435157  0.03835829
    ##   1.0    4.643549e-03  0.06769303  0.1594315  0.03277053
    ##   1.0    1.072720e-02  0.06756953  0.1827182  0.03156821
    ##   1.0    2.478123e-02  0.06857593  0.1116339  0.03247156
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were alpha = 0.3 and lambda = 0.02478123.

Oraz najlepsze prametry:

``` r
model$bestTune
```

    ##    alpha     lambda
    ## 30   0.3 0.02478123

Tabelka jest dość duża, ponieważ mamy aż dwie wartości do dostrojenia.
Najlepsze okazało się wybór \(\alpha = 0.1\), \(\lambda = 0.05285036\).

Teraz dostrójmy model lasów losowych.

``` r
sosenka <- randomForest(formula = Y ~ .,data = data.train)
plot(sosenka)
```

<img src="sosenka.png" width="90%" />

Wykres przedstawia błąd lasu losowego w zależności od liczebności drzew.
Sprawdźmy dla jakiej wartości lasów błąd jest najmniejszy.

``` r
which.min(sosenka$mse)
```

    ## [1] 211

Zatem 211 drzew wydaje się najbardziej opytmalną liczbą. Widzimy
również, że tak naprawdę ta różnica nie jest jakaś strasznie istotna
jeśli liczba drzew przekroczy 200.

Teraz dostroimy resztę parametrów, pownonie korzystając z przeszukwiania
po siadce.

stwórzmy siatkę:

``` r
hyper_grid <- expand.grid(
  mtry       = seq(20, 30, by = 2),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  RMSE   = 0)
```

Sprawdźmy ile łącznie przetestujemy kombinacji:

``` r
nrow(hyper_grid)
```

    ## [1] 96

Teraz pora na proces dostrajania:

``` r
for(i in 1:nrow(hyper_grid)) {
  # train model
  tree <- ranger(
    formula         = Y ~ ., 
    data            = data.train, 
    num.trees       = 211,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
  )
  
  # add OOB error to grid
  hyper_grid$RMSE[i] <- sqrt(tree$prediction.error)
}
```

Wypiszmy te wartości dla których błąd okazał się najmniejszy:

``` r
hyper_grid %>% 
  dplyr::arrange(RMSE) %>%
  head(10)
```

    ##    mtry node_size sampe_size       RMSE
    ## 1    30         3       0.80 0.07807371
    ## 2    20         5       0.70 0.07878721
    ## 3    28         7       0.80 0.07880416
    ## 4    22         7       0.70 0.07885249
    ## 5    28         9       0.80 0.07897604
    ## 6    20         3       0.70 0.07911816
    ## 7    20         5       0.80 0.07915977
    ## 8    20         3       0.55 0.07916800
    ## 9    22         7       0.80 0.07919649
    ## 10   20         7       0.80 0.07920695

AIC

<b> Zadanie 5</b> Widzimy, że najlepiej na zbiorze trenignowym sprawdził
się model elstic net dlatego za pomocą jego przewidzenia zmiennej
objaśnianej.

Na początku nusimy ze zbioru testowego wybrać te same zmienne na których
trenowaliśmy nasz model.

``` r
data.test = subset(data.test, select = -index)
x_test <- as.matrix(data.test)

x <- as.matrix(data.train[,-1])

dim(x_test) #upewnijmy się że wymiary się zgadzają
```

    ## [1] 276 290

Teraz wytrenujemy model.

``` r
main_model <- glmnet(x, y, alpha = 0.1 , lambda = 0.05285036)
```

Ostanim krokiem jest przewidzenie zmiennej objaśnainej dla zbioru
testowego i zapisanie odpowiedniego wektora.

``` r
pred <- main_model %>% predict(x_test) %>% as.vector()

save(pred, file = "Muck.RData")
```
