from main import perceptron_or_and, perceptron_not, perceptron_xor, perceptron_xnor
import pytest


@pytest.mark.parametrize("th0", list(range(-15, -9)))
@pytest.mark.parametrize("th1", list(range(18, 22)))
@pytest.mark.parametrize("th2", list(range(18, 22)))
def test_or(th0, th1, th2):
    assert perceptron_or_and([(0, 0), (0, 1), (1, 0), (1, 1)], [th0, th1, th2]) == [0, 1, 1, 1]


@pytest.mark.parametrize("th0", list(range(-33, -29)))
@pytest.mark.parametrize("th1", list(range(18, 22)))
@pytest.mark.parametrize("th2", list(range(18, 22)))
def test_or(th0, th1, th2):
    assert perceptron_or_and([(0, 0), (0, 1), (1, 0), (1, 1)], [th0, th1, th2]) == [0, 0, 0, 1]


@pytest.mark.parametrize("th0", list(range(1, 5)))
@pytest.mark.parametrize("th1", list(range(-40, -25)))
def test_not(th0, th1):
    assert perceptron_not([0, 1], [th0, th1]) == [1, 0]

@pytest.mark.parametrize("description, theta_or, theta_and, theta_not", [
    ("first", [-15, 18, 19], [-32, 18, 18], [4, -32]),
    ("second", [-10, 18, 18], [-32, 19, 20], [1, -30]),
    ("third", [-14, 18, 22], [-30, 21, 21], [4, -25])])
def test_xor(description, theta_or, theta_and, theta_not):
    assert perceptron_xor([(0, 0), (0, 1), (1, 0), (1, 1)], theta_or, theta_and, theta_not)[0] == [0, 1, 1, 0]


@pytest.mark.parametrize("description, theta_or, theta_and, theta_not", [
    ("forth", [-15, 18, 19], [-32, 18, 18], [4, -32]),
    ("fifth", [-10, 18, 18], [-32, 19, 20], [1, -30]),
    ("sixth", [-14, 18, 22], [-30, 21, 21], [4, -25])])
def test_xnor(description, theta_or, theta_and, theta_not):
    assert perceptron_xnor([(0, 0), (0, 1), (1, 0), (1, 1)], theta_or, theta_and, theta_not)[0] == [1, 0, 0, 1]
