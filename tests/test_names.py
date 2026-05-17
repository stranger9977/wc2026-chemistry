import pytest
from chemistry.names import shorten_name


@pytest.mark.parametrize("full,nick,expected", [
    ("Christian Pulisic", None, "Christian Pulisic"),
    ("Tyler Adams", None, "Tyler Adams"),
    ("Neymar da Silva Santos Junior", None, "Neymar Santos Jr."),
    ("Neymar da Silva Santos Junior", "Neymar", "Neymar"),
    ("Vinícius José Paixão de Oliveira Júnior", "Vinícius Júnior", "Vinícius Júnior"),
    ("Lionel Andrés Messi Cuccittini", "Lionel Messi", "Lionel Messi"),
    ("Lionel Andrés Messi Cuccittini", None, "Lionel Messi"),
    ("Ángel Fabián Di María Hernández", None, "Ángel María"),
    ("Lucas Tolentino Coelho de Lima", "Lucas Paquetá", "Lucas Paquetá"),
    ("Lucas Tolentino Coelho de Lima", None, "Lucas Coelho"),
    ("Raphael Dias Belloli", "Raphinha", "Raphinha"),
])
def test_shorten_name(full, nick, expected):
    assert shorten_name(full, nick) == expected
