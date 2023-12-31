**Синтаксический Анализатор (Parser)**

Программа содержит три основных анализатора:

LL(1) Анализатор: Использует односимвольный просмотр вперед для разбора входных данных в соответствии с заданной грамматикой. Процесс включает в себя вычисление множеств FIRST и FOLLOW, построение таблицы разбора и, наконец, сам процесс разбора.
LL(k) Анализатор: Это расширение LL(1) анализатора, где просмотр вперед увеличивается до k символов. В этом проекте реализован LL(2) вариант.
Анализатор методом рекурсивного спуска: Этот анализатор напрямую использует продукции грамматики для разбора входных данных, вызывая функции для каждого нетерминала.

**Процесс работы**



Подготовка грамматики:
Грамматика считывается из предопределенного текста.
Производится разбор грамматики для дальнейшего использования в анализаторах.
Подготовка к разбору:

Для LL(1) и LL(k) анализаторов вычисляются множества FIRST и FOLLOW.
Строится таблица разбора на основе множеств FIRST и FOLLOW.
Процесс разбора:

Входная строка считывается и передается каждому из анализаторов.
Анализаторы пытаются разобрать входную строку в соответствии с их грамматиками и методиками.
Результаты разбора выводятся на экран.
Визуализация AST: Для некоторых анализаторов также предоставляется визуализация дерева вывода (AST)
