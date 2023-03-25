# 0x05. Advanced Linear Algebra
<h2>Resources</h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=8" title="The determinant | Essence of linear algebra" target="_blank">The determinant | Essence of linear algebra</a></li>
<li><a href="https://www.mathsisfun.com/algebra/matrix-determinant.html" title="Determinant of a Matrix" target="_blank">Determinant of a Matrix</a></li>
<li><a href="https://mathworld.wolfram.com/Determinant.html" title="Determinant" target="_blank">Determinant</a></li>
<li><a href="https://www.quora.com/What-is-the-determinant-of-an-empty-matrix-such-as-a-0x0-matrix" title="Determinant of an empty matrix" target="_blank">Determinant of an empty matrix</a></li>
<li><a href="https://www.youtube.com/watch?v=uQhTuRlWMxw&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=9" title="Inverse matrices, column space and null space" target="_blank">Inverse matrices, column space and null space</a></li>
<li><a href="https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html" title="Inverse of a Matrix using Minors, Cofactors and Adjugate" target="_blank">Inverse of a Matrix using Minors, Cofactors and Adjugate</a></li>
<li><a href="https://mathworld.wolfram.com/Minor.html" title="Minor" target="_blank">Minor</a></li>
<li><a href="https://mathworld.wolfram.com/Cofactor.html" title="Cofactor" target="_blank">Cofactor</a></li>
<li><a href="https://en.wikipedia.org/wiki/Adjugate_matrix" title="Adjugate matrix" target="_blank">Adjugate matrix</a></li>
<li><a href="https://mathworld.wolfram.com/SingularMatrix.html" title="Singular Matrix" target="_blank">Singular Matrix</a></li>
<li><a href="https://stattrek.com/matrix-algebra/elementary-operations" title="Elementary Matrix Operations" target="_blank">Elementary Matrix Operations</a></li>
<li><a href="https://mathworld.wolfram.com/GaussianElimination.html" title="Gaussian Elimination" target="_blank">Gaussian Elimination</a></li>
<li><a href="https://mathworld.wolfram.com/Gauss-JordanElimination.html" title="Gauss-Jordan Elimination" target="_blank">Gauss-Jordan Elimination</a></li>
<li><a href="https://mathworld.wolfram.com/MatrixInverse.html" title="Matrix Inverse" target="_blank">Matrix Inverse</a></li>
<li><a href="https://www.youtube.com/watch?v=PFDu9oVAE-g" title="Eigenvectors and eigenvalues | Essence of linear algebra" target="_blank">Eigenvectors and eigenvalues | Essence of linear algebra</a></li>
<li><a href="https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors" title="Eigenvalues and eigenvectors" target="_blank">Eigenvalues and eigenvectors</a></li>
<li><a href="https://math.mit.edu/~gs/linearalgebra/ila0601.pdf" title="Eigenvalues and Eigenvectors" target="_blank">Eigenvalues and Eigenvectors</a></li>
<li><a href="https://en.wikipedia.org/wiki/Definite_matrix" title="Definiteness of a matrix" target="_blank">Definiteness of a matrix</a> <strong>Up to Eigenvalues</strong></li>
<li><a href="http://mathonline.wikidot.com/definite-semi-definite-and-indefinite-matrices" title="Definite, Semi-Definite and Indefinite Matrices" target="_blank">Definite, Semi-Definite and Indefinite Matrices</a> <strong>Ignore Hessian Matrices</strong></li>
<li><a href="https://www.gaussianwaves.com/2013/04/tests-for-positive-definiteness-of-a-matrix/" title="Tests for Positive Definiteness of a Matrix" target="_blank">Tests for Positive Definiteness of a Matrix</a></li>
<li><a href="https://www.youtube.com/watch?v=tccVVUnLdbc" title="Positive Definite Matrices and Minima" target="_blank">Positive Definite Matrices and Minima</a></li>
<li><a href="https://www.math.utah.edu/~zwick/Classes/Fall2012_2270/Lectures/Lecture33_with_Examples.pdf" title="Positive Definite Matrices" target="_blank">Positive Definite Matrices</a></li>
</ul>
<p><strong>As references</strong>:</p>
<ul>
<li><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html" title="numpy.linalg.eig" target="_blank">numpy.linalg.eig</a></li>
</ul>


## General
* **What is a determinant? How would you calculate it?**

* **What is a minor, cofactor, adjugate? How would calculate them?**
    * **Minor**
    * **cofactor**
    * **adjugate**
* **What is an inverse? How would you calculate it?**

* **What are eigenvalues and eigenvectors? How would you calculate them?**
    * **eigenvalues**

    * **eigenvectors**

* **What is definiteness of a matrix? How would you determine a matrix’s definiteness?**

<div class="panel panel-default" id="project-quiz-questions-title">
    <div class="panel-heading">
      <h3 class="panel-title">
        Quiz questions
      </h3>
    </div>

<div class="panel-body">

<div class="alert alert-info">
    <strong>Great!</strong>
    You've completed the quiz successfully! Keep going!
    <span id="quiz_questions_collapse_toggle">(Show quiz)</span>
</div>

<section class="quiz_questions_show_container" style="display: none;">
    <div class="quiz_question_item_container" data-role="quiz_question874" data-position="1">
    <div class=" clearfix" id="quiz_question-874">

<h4 class="quiz_question">
    
    Question #0
</h4>

<!-- Quiz question Body -->
<p>What is the determinant of the following matrix?</p>

<p>[[  -7,   0,   6 ]<br>
&nbsp; [   5,  -2, -10 ]<br>
&nbsp; [   4,   3,   2 ]]</p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="874">
        <li class="">

  <input type="radio" name="874" id="874-1558208375946" value="1558208375946" data-quiz-answer-id="1558208375946" data-quiz-question-id="874" disabled="disabled" checked="checked">
  <label for="874-1558208375946"><p>-44</p>
</label>
</li>

<li class="">

  <input type="radio" name="874" id="874-1558208376962" value="1558208376962" data-quiz-answer-id="1558208376962" data-quiz-question-id="874" disabled="disabled">
  <label for="874-1558208376962"><p>44</p>
</label>
</li>

<li class="">

  <input type="radio" name="874" id="874-1558208378305" value="1558208378305" data-quiz-answer-id="1558208378305" data-quiz-question-id="874" disabled="disabled">
  <label for="874-1558208378305"><p>14</p>
</label>
</li>

<li class="">

  <input type="radio" name="874" id="874-1558208379666" value="1558208379666" data-quiz-answer-id="1558208379666" data-quiz-question-id="874" disabled="disabled">
  <label for="874-1558208379666"><p>-14</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
<div class="quiz_question_item_container" data-role="quiz_question875" data-position="2">
<div class=" clearfix" id="quiz_question-875">

<h4 class="quiz_question">

Question #1
</h4>

<!-- Quiz question Body -->
<p>What is the minor of the following matrix?</p>

<p>[[  -7,   0,   6 ]<br>
&nbsp; [   5,  -2, -10 ]<br>
&nbsp; [   4,   3,   2 ]]</p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="875">
    <li class="">

  <input type="radio" name="875" id="875-1558208562661" value="1558208562661" data-quiz-answer-id="1558208562661" data-quiz-question-id="875" disabled="disabled">
  <label for="875-1558208562661"><p>[[ 26, 50, 23 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ -18, -38, -21 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 12, 40, 15 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="875" id="875-1558208564998" value="1558208564998" data-quiz-answer-id="1558208564998" data-quiz-question-id="875" disabled="disabled" checked="checked">
  <label for="875-1558208564998"><p>[[ 26, 50, 23 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ -18, -38, -21 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 12, 40, 14 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="875" id="875-1558208572309" value="1558208572309" data-quiz-answer-id="1558208572309" data-quiz-question-id="875" disabled="disabled">
  <label for="875-1558208572309"><p>[[ 26, 50, 23 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ -18, -39, -21 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 12, 40, 14 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="875" id="875-1558208574376" value="1558208574376" data-quiz-answer-id="1558208574376" data-quiz-question-id="875" disabled="disabled">
  <label for="875-1558208574376"><p>[[ 26, 50, 23 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ -18, -39, -21 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 12, 40, 15 ]]</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
<div class="quiz_question_item_container" data-role="quiz_question876" data-position="3">
<div class=" clearfix" id="quiz_question-876">

<h4 class="quiz_question">

Question #2
</h4>

<!-- Quiz question Body -->
<p>What is the cofactor of the following matrix?</p>

<p>[[ 6, -9, 9 ],<br>
&nbsp; [ 7, 5, 0 ],<br>
&nbsp; [ 4, 3, -8 ]]</p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="876">
        <li class="">

  <input type="radio" name="876" id="876-1558209266753" value="1558209266753" data-quiz-answer-id="1558209266753" data-quiz-question-id="876" disabled="disabled">
  <label for="876-1558209266753"><p>[[ -40, 56, 1 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ -45, -84, -54 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ -45, 64, 93 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="876" id="876-1558209268371" value="1558209268371" data-quiz-answer-id="1558209268371" data-quiz-question-id="876" disabled="disabled">
  <label for="876-1558209268371"><p>[[ -40, 56, 1 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ -44, -84, -54 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ -45, 64, 93 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="876" id="876-1558209269413" value="1558209269413" data-quiz-answer-id="1558209269413" data-quiz-question-id="876" disabled="disabled">
  <label for="876-1558209269413"><p>[[ -40, 56, 1 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ -44, -84, -54 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ -45, 63, 93 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="876" id="876-1558209270882" value="1558209270882" data-quiz-answer-id="1558209270882" data-quiz-question-id="876" disabled="disabled" checked="checked">
  <label for="876-1558209270882"><p>[[ -40, 56, 1 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ -45, -84, -54 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ -45, 63, 93 ]]</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
<div class="quiz_question_item_container" data-role="quiz_question877" data-position="4">
<div class=" clearfix" id="quiz_question-877">

<h4 class="quiz_question">

Question #3
</h4>

<!-- Quiz question Body -->
<p>What is the adjugate of the following matrix?</p>

<p>[[ -4, 1, 9 ],<br>
&nbsp; [ -9, -8, -5 ],<br>
&nbsp; [ -3, 8, 10 ]]</p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="877">
        <li class="">

  <input type="radio" name="877" id="877-1558209156651" value="1558209156651" data-quiz-answer-id="1558209156651" data-quiz-question-id="877" disabled="disabled">
  <label for="877-1558209156651"><p>[[ -40, 62, 67 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ 105, -13, -101 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ -97, 29, 41 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="877" id="877-1558209157758" value="1558209157758" data-quiz-answer-id="1558209157758" data-quiz-question-id="877" disabled="disabled">
  <label for="877-1558209157758"><p>[[ -40, 62, 67 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ 105, -14, -101 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ -97, 29, 41 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="877" id="877-1558209158930" value="1558209158930" data-quiz-answer-id="1558209158930" data-quiz-question-id="877" disabled="disabled" checked="checked">
  <label for="877-1558209158930"><p>[[ -40, 62, 67 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 105, -13, -101 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ -96, 29, 41 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="877" id="877-1558209160591" value="1558209160591" data-quiz-answer-id="1558209160591" data-quiz-question-id="877" disabled="disabled">
  <label for="877-1558209160591"><p>[[ -40, 62, 67 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ 105, -14, -101 ],<br>
&nbsp; &nbsp; &nbsp; &nbsp; [ -96, 29, 41 ]]</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

    </div>
    <div class="quiz_question_item_container" data-role="quiz_question878" data-position="5">
    <div class=" clearfix" id="quiz_question-878">

<h4 class="quiz_question">

Question #4
</h4>

<!-- Quiz question Body -->
<p>Is the following matrix invertible? If so, what is its inverse?</p>

<p>[[  1,  0,  1 ]<br>
&nbsp; [  2,  1,  2 ]<br>
&nbsp; [  1,  0, -1 ]]</p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="878">
        <li class="">

  <input type="radio" name="878" id="878-1558207136757" value="1558207136757" data-quiz-answer-id="1558207136757" data-quiz-question-id="878" disabled="disabled">
  <label for="878-1558207136757"><p>[[ 0.5, 0,   0.5 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[    0,  1,     2  ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 0.5,  0,  0.5 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="878" id="878-1558207138051" value="1558207138051" data-quiz-answer-id="1558207138051" data-quiz-question-id="878" disabled="disabled" checked="checked">
  <label for="878-1558207138051"><p>[[ 0.5, 0,   0.5 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[   -2,  1,     0  ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 0.5,  0, -0.5 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="878" id="878-1558207139287" value="1558207139287" data-quiz-answer-id="1558207139287" data-quiz-question-id="878" disabled="disabled">
  <label for="878-1558207139287"><p>[[ 0.5, 0,   0.5 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[    2,  1,     0  ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 0.5,  0,  0.5 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="878" id="878-1558207140581" value="1558207140581" data-quiz-answer-id="1558207140581" data-quiz-question-id="878" disabled="disabled">
  <label for="878-1558207140581"><p>It is singular</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
<div class="quiz_question_item_container" data-role="quiz_question879" data-position="6">
<div class=" clearfix" id="quiz_question-879">

<h4 class="quiz_question">

Question #5
</h4>

<!-- Quiz question Body -->
<p>Is the following matrix invertible? If so, what is its inverse?</p>

<p>[[ 2, 1, 2 ]<br>
&nbsp; [ 1, 0, 1 ]<br>
&nbsp; [ 4, 1, 4 ]]</p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="879">
        <li class="">

  <input type="radio" name="879" id="879-1558207761715" value="1558207761715" data-quiz-answer-id="1558207761715" data-quiz-question-id="879" disabled="disabled">
  <label for="879-1558207761715"><p>[[ 4, 1, 2 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 1, 0, 1 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 4, 1, 2 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="879" id="879-1558207763521" value="1558207763521" data-quiz-answer-id="1558207763521" data-quiz-question-id="879" disabled="disabled">
  <label for="879-1558207763521"><p>[[ 2, 1, 4 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 1, 0, 1 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 2, 1, 4 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="879" id="879-1558207764816" value="1558207764816" data-quiz-answer-id="1558207764816" data-quiz-question-id="879" disabled="disabled">
  <label for="879-1558207764816"><p>[[ 4, 1, 4 ]<br>
&nbsp; &nbsp; &nbsp; &nbsp;[ 1, 0, 1 ]<br>
&nbsp; &nbsp; &nbsp;   [ 2, 1, 2 ]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="879" id="879-1558207766191" value="1558207766191" data-quiz-answer-id="1558207766191" data-quiz-question-id="879" disabled="disabled" checked="checked">
  <label for="879-1558207766191"><p>It is singular</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
<div class="quiz_question_item_container" data-role="quiz_question880" data-position="7">
<div class=" clearfix" id="quiz_question-880">

<h4 class="quiz_question">

Question #6
</h4>

<!-- Quiz question Body -->
<p>Given<br>
<code>A</code> = [[-2, -4, 2],<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [-2, 1, 2],<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [4, 2, 5]]<br>
<code>v</code> = [[2], [-3], [-1]]<br>
Where <code>v</code> is an eigenvector of <code>A</code>, calculate <code>A</code><sup>10</sup><code>v</code></p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="880">
    <li class="">

  <input type="radio" name="880" id="880-1558382082674" value="1558382082674" data-quiz-answer-id="1558382082674" data-quiz-question-id="880" disabled="disabled" checked="checked">
  <label for="880-1558382082674"><p>[[118098], [-177147], [-59049]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="880" id="880-1558382083761" value="1558382083761" data-quiz-answer-id="1558382083761" data-quiz-question-id="880" disabled="disabled">
  <label for="880-1558382083761"><p>[[2097152], [-3145728], [-1048576]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="880" id="880-1558382085222" value="1558382085222" data-quiz-answer-id="1558382085222" data-quiz-question-id="880" disabled="disabled">
  <label for="880-1558382085222"><p>[[2048], [-3072], [-1024]]</p>
</label>
</li>

<li class="">

  <input type="radio" name="880" id="880-1558382086565" value="1558382086565" data-quiz-answer-id="1558382086565" data-quiz-question-id="880" disabled="disabled">
  <label for="880-1558382086565"><p>None of the above</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
<div class="quiz_question_item_container" data-role="quiz_question881" data-position="8">
<div class=" clearfix" id="quiz_question-881">

<h4 class="quiz_question">

Question #7
</h4>

<!-- Quiz question Body -->
<p>Which of the following are also eigenvalues (<code>λ</code>) and eigenvectors (<code>v</code>) of <code>A</code> where<br>
<code>A</code> = [[-2, -4, 2],<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [-2, 1, 2],<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [4, 2, 5]]<br></p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="881">
        <li class="">

  <input type="checkbox" name="881" id="881-1558383730126" value="1558383730126" data-quiz-answer-id="1558383730126" data-quiz-question-id="881" disabled="disabled">
  <label for="881-1558383730126"><p><code>λ</code> = 5; <code>v</code> = [[2], [1], [1]] </p>
</label>
</li>

<li class="">

  <input type="checkbox" name="881" id="881-1558383731627" value="1558383731627" data-quiz-answer-id="1558383731627" data-quiz-question-id="881" disabled="disabled" checked="checked">
  <label for="881-1558383731627"><p><code>λ</code> = -5; <code>v</code> = [[-2], [-1], [1]] </p>
</label>
</li>

<li class="">

  <input type="checkbox" name="881" id="881-1558383733519" value="1558383733519" data-quiz-answer-id="1558383733519" data-quiz-question-id="881" disabled="disabled">
  <label for="881-1558383733519"><p><code>λ</code> = -3; <code>v</code> = [[4], [-2], [3]] </p>
</label>
</li>

<li class="">

  <input type="checkbox" name="881" id="881-1558383735217" value="1558383735217" data-quiz-answer-id="1558383735217" data-quiz-question-id="881" disabled="disabled" checked="checked">
  <label for="881-1558383735217"><p><code>λ</code> = 6; <code>v</code> = [[1], [6], [16]] </p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
<div class="quiz_question_item_container" data-role="quiz_question882" data-position="9">
<div class=" clearfix" id="quiz_question-882">

<h4 class="quiz_question">

Question #8
</h4>

<!-- Quiz question Body -->
<p>What is the definiteness of the following matrix:</p>

<p>[[ -1, 2, 0 ]<br>
&nbsp; [ 2, -5, 2 ]<br>
&nbsp; [ 0, 2, -6 ]]</p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="882">
        <li class="">

  <input type="radio" name="882" id="882-1558384251040" value="1558384251040" data-quiz-answer-id="1558384251040" data-quiz-question-id="882" disabled="disabled">
  <label for="882-1558384251040"><p>Positive definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="882" id="882-1558384252318" value="1558384252318" data-quiz-answer-id="1558384252318" data-quiz-question-id="882" disabled="disabled">
  <label for="882-1558384252318"><p>Positive semi-definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="882" id="882-1558384253882" value="1558384253882" data-quiz-answer-id="1558384253882" data-quiz-question-id="882" disabled="disabled">
  <label for="882-1558384253882"><p>Negative semi-definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="882" id="882-1558384255251" value="1558384255251" data-quiz-answer-id="1558384255251" data-quiz-question-id="882" disabled="disabled" checked="checked">
  <label for="882-1558384255251"><p>Negative definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="882" id="882-1558384256457" value="1558384256457" data-quiz-answer-id="1558384256457" data-quiz-question-id="882" disabled="disabled">
  <label for="882-1558384256457"><p>Indefinite</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
<div class="quiz_question_item_container" data-role="quiz_question883" data-position="10">
<div class=" clearfix" id="quiz_question-883">

<h4 class="quiz_question">

Question #9
</h4>

<!-- Quiz question Body -->
<p>What is the definiteness of the following matrix:</p>

<p>[[ 2, 2, 1 ]<br>
&nbsp; [ 2, 1, 3 ]<br>
&nbsp; [ 1, 3, 8 ]]</p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="883">
<li class="">

  <input type="radio" name="883" id="883-1558384292019" value="1558384292019" data-quiz-answer-id="1558384292019" data-quiz-question-id="883" disabled="disabled">
  <label for="883-1558384292019"><p>Positive definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="883" id="883-1558384293727" value="1558384293727" data-quiz-answer-id="1558384293727" data-quiz-question-id="883" disabled="disabled">
  <label for="883-1558384293727"><p>Positive semi-definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="883" id="883-1558384303928" value="1558384303928" data-quiz-answer-id="1558384303928" data-quiz-question-id="883" disabled="disabled">
  <label for="883-1558384303928"><p>Negative semi-definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="883" id="883-1558384311647" value="1558384311647" data-quiz-answer-id="1558384311647" data-quiz-question-id="883" disabled="disabled">
  <label for="883-1558384311647"><p>Negative definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="883" id="883-1558384318860" value="1558384318860" data-quiz-answer-id="1558384318860" data-quiz-question-id="883" disabled="disabled" checked="checked">
  <label for="883-1558384318860"><p>Indefinite</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
        <div class="quiz_question_item_container" data-role="quiz_question884" data-position="11">
    <div class=" clearfix" id="quiz_question-884">

<h4 class="quiz_question">
        
    Question #10
</h4>

<!-- Quiz question Body -->
<p>What is the definiteness of the following matrix:</p>

<p>[[ 2, 1, 1 ]<br>
&nbsp; [ 1, 2, -1 ]<br>
&nbsp; [ 1, -1, 2 ]]</p>


<!-- Quiz question Answers -->
<ul class="quiz_question_answers" data-question-id="884">
        <li class="">

  <input type="radio" name="884" id="884-1558384325145" value="1558384325145" data-quiz-answer-id="1558384325145" data-quiz-question-id="884" disabled="disabled">
  <label for="884-1558384325145"><p>Positive definite</p>
</label>
</li>

 <li class="">

  <input type="radio" name="884" id="884-1558384330459" value="1558384330459" data-quiz-answer-id="1558384330459" data-quiz-question-id="884" disabled="disabled" checked="checked">
  <label for="884-1558384330459"><p>Positive semi-definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="884" id="884-1558384336616" value="1558384336616" data-quiz-answer-id="1558384336616" data-quiz-question-id="884" disabled="disabled">
  <label for="884-1558384336616"><p>Negative semi-definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="884" id="884-1558384352805" value="1558384352805" data-quiz-answer-id="1558384352805" data-quiz-question-id="884" disabled="disabled">
  <label for="884-1558384352805"><p>Negative definite</p>
</label>
</li>

<li class="">

  <input type="radio" name="884" id="884-1558384369570" value="1558384369570" data-quiz-answer-id="1558384369570" data-quiz-question-id="884" disabled="disabled">
  <label for="884-1558384369570"><p>Indefinite</p>
</label>
</li>

</ul>

<!-- Quiz question Tips -->

</div>

</div>
</section>
</div>
</div>

# Tasks

## 0. Determinant
Write a function `def determinant(matrix):` that calculates the determinant of a matrix:

* `matrix` is a list of lists whose determinant should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If matrix is not square, raise a ValueError with the message matrix must be a square matrix
* The list `[[]]` represents a `0x0` matrix
* Returns: the determinant of `matrix`

```
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    determinant = __import__('0-determinant').determinant

    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(determinant(mat0))
    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat4))
    try:
        determinant(mat5)
    except Exception as e:
        print(e)
    try:
        determinant(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./0-main.py 
1
5
-2
0
192
matrix must be a list of lists
matrix must be a square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
* File: [0-determinant.py]()

## 1. Minor
Write a function `def minor(matrix):` that calculates the minor matrix of a matrix:

* `matrix` is a list of lists whose minor matrix should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
* Returns: the minor matrix of `matrix`

```
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    minor = __import__('1-minor').minor

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(minor(mat1))
    print(minor(mat2))
    print(minor(mat3))
    print(minor(mat4))
    try:
        minor(mat5)
    except Exception as e:
        print(e)
    try:
        minor(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./1-main.py 
[[1]]
[[4, 3], [2, 1]]
[[1, 1], [1, 1]]
[[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
* File: [1-minor.py]()

## 2. Cofactor
Write a function `def cofactor(matrix):` that calculates the cofactor matrix of a matrix:

* `matrix` is a list of lists whose minor matrix should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
* Returns: the minor matrix of `matrix`

```
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    cofactor = __import__('2-cofactor').cofactor

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(cofactor(mat1))
    print(cofactor(mat2))
    print(cofactor(mat3))
    print(cofactor(mat4))
    try:
        cofactor(mat5)
    except Exception as e:
        print(e)
    try:
        cofactor(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./2-main.py 
[[1]]
[[4, -3], [-2, 1]]
[[1, -1], [-1, 1]]
[[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
* File: [2-cofactor.py]

## 3. Adjugate
Write a function `def adjugate(matrix):` that calculates the adjugate matrix of a matrix:

* `matrix` is a list of lists whose minor matrix should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
* Returns: the minor matrix of `matrix`

```
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    adjugate = __import__('3-adjugate').adjugate

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(adjugate(mat1))
    print(adjugate(mat2))
    print(adjugate(mat3))
    print(adjugate(mat4))
    try:
        adjugate(mat5)
    except Exception as e:
        print(e)
    try:
        adjugate(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./3-main.py 
[[1]]
[[4, -2], [-3, 1]]
[[1, -1], [-1, 1]]
[[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```

## 4. Inverse
Write a function def inverse(matrix): that calculates the inverse of a matrix:

* `matrix` is a list of lists whose minor matrix should be calculated
* If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
* If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
* Returns: the inverse of `matrix`, or `None` if matrix is singular
```
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 4-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    inverse = __import__('4-inverse').inverse

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(inverse(mat1))
    print(inverse(mat2))
    print(inverse(mat3))
    print(inverse(mat4))
    try:
        inverse(mat5)
    except Exception as e:
        print(e)
    try:
        inverse(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./4-main.py 
[[0.2]]
[[-2.0, 1.0], [1.5, -0.5]]
None
[[-0.0625, -0.052083333333333336, 0.24479166666666666], [0.1875, -0.17708333333333334, -0.06770833333333333], [0.0, 0.16666666666666666, -0.08333333333333333]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
* File: [4-inverse.py]

## 5. Definiteness
Write a function `def definiteness(matrix):` that calculates the definiteness of a matrix:

* `matrix` is a `numpy.ndarray` of shape `(n, n)` whose definiteness should be calculated
* If matrix is not a `numpy.ndarray`, raise a `TypeError` with the message `matrix must be a numpy.ndarray`
* If `matrix` is not a valid matrix, return `None`
* Return: the string `Positive definite, Positive semi-definite, Negative semi-definite, Negative definite,` or `Indefinite` if the matrix is positive definite, positive semi-definite, negative semi-definite, negative definite of indefinite, respectively
* If matrix does not fit any of the above categories, return None
* You may `import numpy as np`

```
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 5-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    definiteness = __import__('5-definiteness').definiteness
    import numpy as np

    mat1 = np.array([[5, 1], [1, 1]])
    mat2 = np.array([[2, 4], [4, 8]])
    mat3 = np.array([[-1, 1], [1, -1]])
    mat4 = np.array([[-2, 4], [4, -9]])
    mat5 = np.array([[1, 2], [2, 1]])
    mat6 = np.array([])
    mat7 = np.array([[1, 2, 3], [4, 5, 6]])
    mat8 = [[1, 2], [1, 2]]

    print(definiteness(mat1))
    print(definiteness(mat2))
    print(definiteness(mat3))
    print(definiteness(mat4))
    print(definiteness(mat5))
    print(definiteness(mat6))
    print(definiteness(mat7))
    try:
        definiteness(mat8)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./5-main.py 
Positive definite
Positive semi-definite
Negative semi-definite
Negative definite
Indefinite
None
None
matrix must be a numpy.ndarray
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```