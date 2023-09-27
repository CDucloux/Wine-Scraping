var cities = new Array('monaco', 'monte carlo', 'monte-carlo', 'montecarlo');

//lorsque la ville est modifiée
$(document).on('keyup','#city', function(event) {
    city = $('#city').val().toLowerCase();

    //si la ville entrée fait partie du tableau "cities" (inArray retourne -1 si la condition est fausse)
    if ($.inArray (city, cities) > -1) {
        //on impose le Pays Monaco
        $('#id_country option').attr('selected', false);
        $('#id_country option[value="8"]').attr('selected', true);
        $('#uniform-id_country span').text('France');
        $('#id_country').prop('disabled', true);
    }
    else {
        //sinon on libère le select du Pays
        $('#id_country').prop('disabled', false);
    }
});
