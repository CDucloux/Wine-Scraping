(function() {

  function insertScript(content, type, id) {
    var s = document.createElement('script');
    var body = document.querySelector('body');
    s.type = type;
    if (id) {
      s.id = id;
    }
    s.innerText = content;
    body.appendChild(s);
  }

  function insertCSS(content) {
    var sheet = document.createElement('style');
    var head = document.querySelector('head');
    sheet.type = 'text/css';

    sheet.innerText = content;
    head.appendChild(sheet);
  }






}());

var doofinder_script ='//cdn.doofinder.com/media/js/doofinder-classic.7.latest.min.js';
(function(d,t){var f=d.createElement(t),s=d.getElementsByTagName(t)[0];f.async=1;
f.src=('https:'==location.protocol?'https:':'http:')+doofinder_script;
f.setAttribute('charset','utf-8');
s.parentNode.insertBefore(f,s)}(document,'script'));

var HASHID= "9c8ae9f95095359dd7735b114464a192";
var dfClassicLayers = [{
  "queryInput": "#search_query_top",
  "hashid": HASHID,
  "zone": "eu1",  
  "searchParams": {
    "type": ["product", "domaines"]
  },
  "mobile": {
      "display": {    
    "results": { 
        "template": document.getElementById('results_template').innerHTML.replace(/;/g,"")       
       }
        }      
  },  
  "display": {
    "lang": "fr",
    "template": document.getElementById('container_template').innerHTML.replace(/;/g,""),
    "results": {
      "template": document.getElementById('results_template').innerHTML.replace(/;/g,"")       
     },
    "translations": {
      "CLEAR": "SUPPRIMER",
      "CLOSE": "VALIDER",
      "FILTER": "FILTRER",
      "Rated by": "Noté par",
      "customer(s)": "client(s)",
      "from": "à partir de",  
    },
    "closeOnClick": true,
    "closeOnEscKey": true,   
    templateFunctions: {
      LabelAsImage: function() {
        return function(text, render) {         
          var labels = render(text).split("/");        
          for (var i=0; i< labels.length; i++) {          
              if (labels[i].trim().match(/bio/i)) {
                return "<p class='color-green top-label-shadow-green'>BIO<i class='vin-picto-bio'></i></p>";
              }                       
           }
        }
      },
      RewardAsImage: function() {
        return function(text, render) {         
          var rewards = render(text).split(";");
          var images = "";                  
          for (var i=0; i<rewards.length; i++) {                     
               images = images.concat("<img class='df-rewards' src='/img/recompenses/");
               images  = images.concat(rewards[i]);
               images  = images.concat("'/>");                     
           }
          return images;        
        }
       },    
      FormatedCustomer : function() {
        return function(text, render) {
          if (render(text) > 1) {
            return "clients";
          }
          else {
            return "client";
          }          
        }
      },
      
    }                  
  },
  "callbacks":{
      "loaded": function (instance){
          var commonClient = new doofinder.core.Client(HASHID, {zone: 'eu1'});
	      var commonInputWidget = new doofinder.core.widgets.QueryInput("#search_query_top");

          var urlsController = new doofinder.core.Controller(commonClient, {rpp: 5, type: 'urls'});
          var urlsResultsWidget = new doofinder.core.widgets.ScrollDisplay("#df-urls-container", {
		      offset: 200,
		      template: document.getElementById('df-urls-template').innerHTML.replace(/;/g,""),
              templateFunctions: {
                CheckIsBlog : function() {
                  return function(text, render) {
                      if (render(text) == 'true') {
                          return ("<span class='df-blog__suggest'>Blog</span>");
                      }
                  }
                },
                PurpleColor : function() {
                  return function(text, render) {
                      if (render(text) == 'true') {
                          return ("color-purple-dark text-lowercase");
                      } else {
                          return ("text-uppercase");
                      }
                  }
                } 
              }
	      });
          urlsController.search(instance.layer.controller.query);
	      urlsController.registerWidgets([commonInputWidget, urlsResultsWidget]);          
      }
  },
  "googleAnalytics": true
}];
