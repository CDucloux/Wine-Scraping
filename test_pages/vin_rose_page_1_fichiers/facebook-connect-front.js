/**
* 2012 - 2020 HiPresta
*
* MODULE Facebook Connect
*
* @author    HiPresta <support@hipresta.com>
* @copyright HiPresta 2020
* @license   Addons PrestaShop license limitation
* @link      https://hipresta.com
*
* NOTICE OF LICENSE
*
* Don't use this module on several shops. The license provided by PrestaShop Addons
* for all its modules is valid only once for a single shop.
*/

function loaderOpening (){
	$('body').append('<div class="hi_spinner"><img src="'+sc_fb_loader+'"></div>');
	var top = $('body').scrollTop();
	$('.hi_spinner').css('top', top+'px');
	$('body').addClass('pf_overflow');
}
function loaderClose (){
	$('.hi_spinner').hide();
	$('body').removeClass('pf_overflow');
}

/*Facebook login start*/
function FbLogin() {
	FB.api('/me?fields=email,first_name,last_name,gender', function(response) {
		$.ajax({
			type: "POST",
			dataType: "json",
			url: hi_sc_fb_front_controller_dir,
			data: {
				action : 'get_facebook_info',
				user_fname: response.first_name,
				user_lname: response.last_name,
				email: response.email,
				user_data_id: response.id,
				gender: response.gender
			},
			beforeSend: function(){
				loaderOpening();
			},
			success: function(response){
				if(response.activate_die_url != ''){
					 window.location.href = response.activate_die_url;
				} else {
					if(fb_connect_back) {
						window.location.href = fb_connect_back;
					} else if (redirect == 'authentication_page') {
						window.location.href = authentication_page;
					} else {
						location.reload();
					}
				}
			}
		});
	});
}

function fb_login(e){
	FB.login(function(response) {
		if (response.authResponse) {
			access_token = response.authResponse.accessToken;
			user_id = response.authResponse.userID;
			FbLogin();
		}
	},
	{
		scope: 'public_profile,email'
	});
}
/*Facebook login end*/

$(document).ready(function() {
	$(".onclick-btn").click(function(){
		return false;
	});
});
