$(document).ready(function(){
   $('span', 'nav#menu_gauche').live('click', function(e){
        var nav_value = '';
        var $el = $(e.target);

        if ($el.parents('li').length) {
            nav_value = $el.closest('section').find('a').text().trim() + ' / ';
        }
        nav_value += $el.text();

        try {
            tc_events_1(e, 'navigationClick', {'eventCategory': 'Navigation', 'eventAction': 'Menu Gauche', 'eventLabel': nav_value});
            // gtag('event', 'Navigation', {
            //   'event_category': 'MenuGauche',
            //   'event_label': onav_value
            // });
        } catch { }
        // console.log(nav_value);
    });

   $('ul.mnnv').live('click', function(e){
        var nav_value = [];
        var $el = $(e.target);
        if($el.is('a') || $el.is('img')) {

            if(!$el.is('.mnnv-tb') && $el.parents('li.nav-tab-vinatis').children('a.mnnv-tb:first').length) {
                nav_value.push($el.parents('li.nav-tab-vinatis').children('a.mnnv-tb:first').text().trim());
            }
            if ($el.parents('.mnnv-cl').children('h3:first').length) {
                nav_value.push($el.parents('.mnnv-cl').children('h3:first').text().trim());
            }
            if($el.is('a')) {
                nav_value.push($el.text());
            } else {
                var image_label = 'Image';
                if ($el.attr('alt')) {
                    image_label += ' (' + $el.attr("alt") + ')';
                }
                nav_value.push(image_label);
            }

            // console.log(nav_value.join(' / '));
            try {
                tc_events_1(e, 'navigationClick', {'eventCategory': 'Navigation', 'eventAction': 'Menu Haut', 'eventLabel': nav_value});
            } catch { }
        }
    })
});
