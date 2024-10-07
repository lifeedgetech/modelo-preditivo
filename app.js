<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Login</title>
    <link rel="stylesheet" type="text/css" href="css/style.css">
    <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.8.2/css/all.css"
        integrity="sha384-oS3vJWv+0UjzBfQzYUhtDYW+Pj2yciDJxpsK1OYPAYjqT085Qq/1cq5FLXAZQ7Ay" crossorigin="anonymous">
</head>
<body>
    <div class="container">
        <div class="content first-content">
            <div class="first-column">
                <h2 class="title title-primary">Bem vindo de volta!</h2>
                <p class="description description-primary">Para ficar conectado com a gente</p>
                <p class="description description-primary">por favor faça o login com o seu CRM</p>
                <button id="signin" class="btn btn-primary">entrar</button>
            </div>    
            <div class="second-column">
                <h2 class="title title-second">Crie sua conta</h2>
                <p class="description description-second">use seu CRM abaixo:</p>
                <form class="form">
                    <label class="label-input" for="">
                        <i class="far fa-user icon-modify"></i>
                        <input type="text" placeholder="Nome">
                    </label>
                    
                    <label class="label-input" for="">
                        <i class="fas fa-stethoscope icon-modify"></i>
                        <input type="email" placeholder="CRM">
                    </label>
                    
                    <label class="label-input" for="">
                        <i class="fas fa-lock icon-modify"></i>
                        <input type="password" placeholder="Senha">
                    </label>
                </form>
                <a href="upload.html" >
                    <button class="btn btn-second">criar</button>
                </a>
                    
            </div><!-- second column -->
        </div><!-- first content -->
        <div class="content second-content">
            <div class="first-column">
                <h2 class="title title-primary">Olá, Doutor(a)!</h2>
                <p class="description description-primary">Bem Vindo!</p>
                <button id="signup" class="btn btn-primary">criar conta</button>
            </div>
            <div class="second-column">
                <h2 class="title title-second">Acesse</h2>
                <p class="description description-second">Entre com seu CRM:</p>
                <form class="form">
                
                    <label class="label-input" for="">
                        <i class="fas fa-stethoscope icon-modify"></i>
                        <input type="email" placeholder="CRM">
                    </label>
                
                    <label class="label-input" for="">
                        <i class="fas fa-lock icon-modify"></i>
                        <input type="password" placeholder="Senha">
                    </label>
                
                    <a class="password" href="#">esqueceu sua senha?</a>
                </form>
                <a href="upload.html" >
                    <button class="btn btn-second">entrar</button>
                </a>
            </div><!-- second column -->
        </div><!-- second-content -->
    </div>
    <script src="js/app.js"></script>
</body>
</html>